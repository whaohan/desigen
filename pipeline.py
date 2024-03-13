import os
import argparse
import random
import torch
import torchvision.transforms.functional as F
from background.generator import get_bg_generator
from layout.dataset import get_dataset
from layout.decoder import GPT, GPTConfig
from layout.encoder import get_encoder
from layout.trainer import TrainerConfig, Eval
from layout.utils import seq_to_bbox, set_seed, layout_to_mask
from PIL import Image
from background import text2image
from saliency.basnet import get_saliency_model, saliency_detect
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str)
parser.add_argument("--mode", choices=["background", "design", "iteration"], default="background")
parser.add_argument("--iteration", type=int, default=1)
parser.add_argument("--dataset", choices=["webui"], default="webui", const='bbox',nargs='?')
parser.add_argument("--data_dir", default="data/processed", help="/path/to/dataset/dir")
parser.add_argument("--log_dir", default="logs/pipeline", help="/path/to/logs/dir")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--encoder_path", type=str, default='logs/layout-swin/encoder_99.pth')
parser.add_argument("--decoder_path", type=str, default='logs/layout-swin/decoder_99.pth')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--layout_num", type=int, default=8)
parser.add_argument("--encode_backbone", type=str, default='swin')
parser.add_argument('--encode_embd', default=1024, type=int)
parser.add_argument('--n_layer', default=6, type=int)
parser.add_argument('--n_embd', default=512, type=int)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--mask_image', type=str, default=None)
args = parser.parse_args()


def get_evaler():
    print(f"using device: {device}")
    print("train dataset vocab_size: ", train_dataset.vocab_size)
    print("train dataset max_length: ", train_dataset.max_length)
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      encode_embd=args.encode_embd,
                      component_class=train_dataset.component_class)  # a GPT-1
    decoder_model = GPT(mconf)
    encoder_model = get_encoder(name=args.encode_backbone, pretrain=False)
    encoder_model.load_state_dict(torch.load(args.encoder_path))
    decoder_model.load_state_dict(torch.load(args.decoder_path))
    tconf = TrainerConfig(dataset=args.dataset,
                          samples_dir=samples_dir,
                          encoder_path=args.encoder_path,
                          decoder_path=args.decoder_path,
                          device=args.device)

    evaler = Eval(encoder_model, decoder_model, train_dataset, tconf)
    return evaler


log_dir = args.log_dir
samples_dir = os.path.join(log_dir, "evaluate_samples")
set_seed(args.seed)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
train_dataset = get_dataset(args.dataset, "train", args.data_dir)
evaler = get_evaler()
saliency_model = get_saliency_model()


def max_mask(layouts):
    max_sum = 0
    for layout in layouts:
        cur_mask = layout_to_mask(layout, evaler.bos_token, evaler.eos_token, evaler.pad_token)
        if cur_mask.sum() > max_sum:
            mask = cur_mask
            max_sum = cur_mask.sum()
    return mask


def occlusion(saliency, layout):
    res, area = 0, 0
    _, boxes = seq_to_bbox(layout, train_dataset.bos_token, train_dataset.eos_token, train_dataset.pad_token)
    for box in boxes:
        res += saliency[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].sum()
        area += (box[2] * box[3]).item()
    return res / area


def optimize_layout(image, layouts):
    best_occ = 1
    best_layout = None
    saliency = saliency_detect(saliency_model, image, threshold=1).squeeze()
    if len(layouts) == 0:
        return layouts
    for _, layout in enumerate(layouts):
        occ = occlusion(saliency, layout)
        if occ < best_occ:
            best_occ = occ
            best_layout = layout
    
    return best_layout
        

model = get_bg_generator(mask_image=args.mask_image, degrate=0.001)


def generate_design(prompt):
    sample_dir = os.path.join(log_dir, 'design')
    images = model(prompt=[prompt] * 8).images
    cnt = len(os.listdir(sample_dir))
    os.makedirs(os.path.join(sample_dir, str(cnt)))
    for i in range(len(images)):
        image = images[i]
        image_tensor = F.to_tensor(image.resize((224, 224))).unsqueeze(0).to(args.device)
        # generate layout
        category = ['text'] * random.randint(2, 4) + ['button'] * random.randint(0, 2)
        layouts = evaler.generate(image_tensor, saliency=None, category=category, generated_num=args.layout_num)
        # render layout
        for t in range(len(layouts)):
            layout_img = train_dataset.render(layouts[t], image.copy(), W=512, H=512, border=0)
            layout_img.save(os.path.join(sample_dir, str(cnt), f'layout_{i}_{t}.png'))


def generate_background(prompt):
    save_dir = os.path.join(log_dir, 'background')
    images = model(prompt=[prompt] * 16).images
    cnt = len(os.listdir(save_dir))
    os.makedirs(os.path.join(save_dir, str(cnt)))
    for i in range(len(images)):
        images[i].save(os.path.join(save_dir, str(cnt), f'background_{i}.png'))


def iterative_refine(prompt):
    save_dir = 'iterative_refine'
    category = ['text'] * random.randint(2, 4) + ['button'] * random.randint(0, 2)
    iteration = 3
    
    os.makedirs(os.path.join(log_dir, save_dir), exist_ok=True)
    cnt = len(os.listdir(os.path.join(log_dir, save_dir)))
    samples_dir = os.path.join(log_dir, save_dir, str(cnt))
    os.makedirs(samples_dir, exist_ok=True)
    layout_list = []
    
    # get background image
    images, latent = text2image(prompt=[prompt])
    image = Image.fromarray(images[0])
    image.save(os.path.join(samples_dir, f'background_0.png'))    
    image_tensor = F.to_tensor(image.resize((224, 224))).unsqueeze(0).to(args.device)
    layouts = evaler.generate(image_tensor, saliency=None, category=category, generated_num=args.layout_num)
    opt_layout = optimize_layout(image_tensor, layouts)
    layout_img = train_dataset.render(opt_layout.copy(), image.copy(), W=512, H=512)
    layout_img.save(os.path.join(samples_dir, f'layout_0.png'))
    box_and_label = train_dataset.render_normalized_layout(opt_layout.copy())
    layout_list.append(box_and_label)
    
    mask_layout = layout_to_mask(opt_layout, evaler.bos_token, evaler.eos_token, evaler.pad_token)
    mask = torch.zeros(mask_layout.shape, device=model.device).unsqueeze(0)
    
    for iter in range(iteration):
        mask = torch.logical_or(torch.tensor(mask_layout, device=model.device).unsqueeze(0), mask)
        image, latent = text2image(prompt=[prompt], no_attn_mask=mask, degrate=0.1, latent=latent)
        image = Image.fromarray(image[0])
        image.save(os.path.join(samples_dir, f'background_{1+iter}.png'))  
        image_tensor = F.to_tensor(image.resize((224, 224))).unsqueeze(0).to(args.device)
        layouts = evaler.generate(image_tensor, saliency=None, category=category, generated_num=args.layout_num)
        opt_layout = optimize_layout(image_tensor, layouts)
        mask_layout = layout_to_mask(opt_layout.copy(), evaler.bos_token, evaler.eos_token, evaler.pad_token)        
        layout_img = train_dataset.render(opt_layout.copy(), image.copy(), W=512, H=512)
        layout_img.save(os.path.join(samples_dir, f'layout_{1 + iter}.png'))   
        box_and_label = train_dataset.render_normalized_layout(opt_layout.copy())
        layout_list.append(box_and_label)

    with open(os.path.join(samples_dir, f'layouts.pkl'), 'wb') as fb:
        pickle.dump(layout_list, fb)


if __name__ == "__main__":
    if args.mode == 'background':
        generate_background(prompt = args.prompt)
    elif args.mode == 'design':
        generate_design(prompt = args.prompt)
    elif args.mode == 'iteration':
        iterative_refine(prompt=args.prompt)
