import argparse
import json
import os

import torch
import torchvision.transforms.functional as T
from diffusers import StableDiffusionPipeline
from einops import rearrange
from PIL import Image
from bg_utils import add_noise, latent2image, set_seeds

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_name_or_path",
    type=str,
    default="../logs/background",
)
parser.add_argument(
    "--save_path",
    type=str,
    default='validation/ours',
)
parser.add_argument(
    "--val_path",
    type=str,
    default='../data/background/val/metadata.jsonl',
)
args =parser.add_argument(
    "--seed",
    type=int,
    default=11,
) 
args =parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
) 
args = parser.parse_args()


# load model
model_path = args.pretrained_model_name_or_path
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

seed = args.seed
set_seeds(seed)
print('using seed:', seed)
print('using model:', model_path)

# load validation data
val_path = args.val_path
save_dir = args.save_path
os.makedirs(save_dir, exist_ok=True)
batch_size = args.batch_size


def inj_forward(degrate=1, no_attn_mask=None):
    def forward(self, x, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, dim = x.shape
        h = self.heads
        q = self.to_q(x)  # (batch_size, 64*64, 320)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if is_cross else x
        k = self.to_k(encoder_hidden_states) # (batch_size, 77, 320)
        v = self.to_v(encoder_hidden_states)
        q = self.head_to_batch_dim(q)
        k = self.head_to_batch_dim(k)
        v = self.head_to_batch_dim(v)

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~attention_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        # attention degration
        if is_cross:
            attn = rearrange(attn, 'b (h w) t -> b t h w', h=int((attn.shape[1])**0.5))
            if degrate != 1 and no_attn_mask is not None:
                cur_mask = T.resize(no_attn_mask, attn.shape[2:]).bool().squeeze()
                attn[:, :, cur_mask] *= degrate
            attn = rearrange(attn, 'b t h w -> b (h w) t')
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = self.batch_to_head_dim(out)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
    return forward


@torch.no_grad()
def text2image(prompt, latent=None, negative_prompt=None, strength=0.8, guidance_scale=7.5, height=512, width=512):
    global attn_store
    batch_size = len(prompt)
    if latent is None:
        latent = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8))
        t_start = 0
    else:
        latent, t_start = add_noise(latent, pipe.scheduler, strength=strength)
    # encode prompt embeddings
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        negative_prompt if negative_prompt is not None else ([''] * batch_size), 
        padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    latents = latent.expand(batch_size, pipe.unet.in_channels, height // 8, width // 8).to(pipe.device)
    latents = latents.to(text_embeddings.dtype)
    
    pipe.scheduler.set_timesteps(num_inference_steps=50)
    for t in pipe.scheduler.timesteps[t_start:]:
        input_latents = torch.cat([latents] * 2)
        input_latents = pipe.scheduler.scale_model_input(input_latents, t)
        noise_pred = pipe.unet(input_latents, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    image = latent2image(pipe.vae, latents)
    return image


# generate validation set examples
items = []
cnt = 0
with open(val_path, 'r') as f:
    for item in f:
        cnt += 1
        item = json.loads(item)
        items.append(item)
        if len(items) == batch_size or cnt == 1000:
            prompts = ['A Background image of ' + item['text'] for item in items]
            images = text2image(prompt=prompts, negative_prompt=None)
            for i in range(len(images)):
                Image.fromarray(images[i]).save(os.path.join(save_dir, items[i]['file_name']))
            items = []

