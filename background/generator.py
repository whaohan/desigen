import torch
from PIL import Image
from .bg_utils import latent2image, set_seeds
import torchvision.transforms.functional as T
from diffusers import StableDiffusionPipeline
from einops import rearrange

SEED = 888
RES = 16
DEGRATE = 1
set_seeds(SEED)
torch.set_grad_enabled(False)
device = torch.device('cuda:0')
model = StableDiffusionPipeline.from_pretrained("logs/background", safety_checker=None).to(device)
model.scheduler.set_timesteps(num_inference_steps=50)
attn_store = []
avg_store = []



def inj_forward(degrate=DEGRATE, no_attn_mask=None):
    
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
        if is_cross:
            attn = rearrange(attn, 'b (h w) t -> b t h w', h=int((attn.shape[1])**0.5))
            if degrate != 1 and no_attn_mask is not None:
                cur_mask = T.resize(no_attn_mask, attn.shape[2:]).bool().squeeze()
                attn[:, :, cur_mask] *= degrate
            # if attn.shape[2] == RES:
            #     attn_store.append(attn)  # ([16, 77, RES, RES])
            attn = rearrange(attn, 'b t h w -> b (h w) t')
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = self.batch_to_head_dim(out)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out
    
    return forward


def get_bg_generator(mask_image=None, degrate=0.1):
    global model
    # model = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None).to(device)
    if mask_image is not None:
        print(f'use mask in {mask_image}')
        mask_image = T.to_tensor(Image.open(mask_image).convert('L'))
        for _module in model.unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward(degrate=degrate, no_attn_mask=mask_image)
    return model


def visualize_attn_map(prompt, res=RES):
    # visualize the attn map
    # best visualization in 16 * 16 resolution
    global avg_store, attn_store
    b, l, _, _ = attn_store[0].shape
    avg = torch.zeros(b, l, res, res, device=attn_store[0].device)
    for attn in attn_store:
        if attn.shape[-1] == res:
            avg += attn.squeeze(0)
    avg /= len(attn_store)
    avg_store.append(avg.unsqueeze(0))
    # show_cross_attention(avg.mean(0).cpu(), prompt[0], model.tokenizer, name=f'attn/attention_{len(avg_store)}')


@torch.no_grad()
def text2image(prompt, latent=None, negative_prompt=None, strength=0.8, guidance_scale=7.5, height=512, width=512, no_attn_mask=None, degrate=1):
    if no_attn_mask != None:
        for _module in model.unet.modules():
            if _module.__class__.__name__ == "CrossAttention":
                _module.__class__.__call__ = inj_forward(degrate=degrate, no_attn_mask=no_attn_mask)
    global attn_store
    batch_size = len(prompt)
    if latent is None:
        latent = torch.randn((1, model.unet.in_channels, height // 8, width // 8))
        t_start = 0
    else:
        # latent, t_start = add_noise(latent, model.scheduler, strength=strength)
        t_start = 0
    # encode prompt embeddings
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        negative_prompt if negative_prompt is not None else ([''] * batch_size), 
        padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    latents = latents.to(text_embeddings.dtype)
    
    model.scheduler.set_timesteps(num_inference_steps=50)
    for t in model.scheduler.timesteps[t_start:]:
        input_latents = torch.cat([latents] * 2)
        input_latents = model.scheduler.scale_model_input(input_latents, t)
        noise_pred = model.unet(input_latents, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    
    image = latent2image(model.vae, latents)
    return image, latent

