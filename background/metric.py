import glob
import json
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore
import os, re
from PIL import Image
import sys
sys.path.append('..')

_ = torch.manual_seed(123)
name2text = {}
fid = FrechetInceptionDistance(feature=192).cuda()
clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()

def cal_fid(img1, img2):
    img1_ = (img1 * 255).type(torch.uint8)
    img2_ = (img2 * 255).type(torch.uint8)
    fid.update(img1_, real=True)
    fid.update(img2_, real=False)

def cal_clip(img, captions):
    img_ = (img * 255).type(torch.uint8)
    clip(img_, captions)


def main(src_dir):
    dst_dir = '../data/background/val'
    meta_path = f'{dst_dir}/metadata.jsonl'
    files = os.listdir(src_dir)
    batch_size = 36
    src_img, dst_img, captions = [], [], []
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            txt = re.split('[,.\- ]', item['text'])
            name2text[item['file_name']] = ' '.join(txt)
    
    for name in files:
        src_path = os.path.join(src_dir, name)
        dst_path = os.path.join(dst_dir, name)
        src_img.append(torchvision.transforms.functional.to_tensor(Image.open(src_path)).unsqueeze(0))
        dst_img.append(torchvision.transforms.functional.to_tensor(Image.open(dst_path)).unsqueeze(0))
        captions.append(name2text[name])
        
        if len(src_img) == batch_size or name == files[-1]:
            src_tensor = torch.cat(src_img).cuda()
            dst_tensor = torch.cat(dst_img).cuda()
            cal_fid(src_tensor, dst_tensor)
            try:
                cal_clip(src_tensor, captions)
            except:
                pass
            src_img, dst_img, captions = [], [], []

    
    print('Metric between:', src_dir, dst_dir)
    print('\tfid↓: %.2f' % fid.compute().item())
    print('\tclip↑: %.3f' % clip.compute().item())


def cal_saliency(src_dir):
    from saliency.basnet import get_saliency_model, saliency_detect
    saliency_model = get_saliency_model()
    files = os.listdir(src_dir)
    batch_size = 32
    img = []
    res = []
    
    for name in files:
        src_path = os.path.join(src_dir, name)
        img.append(torchvision.transforms.functional.to_tensor(Image.open(src_path).resize((224, 224))))
        
        if len(img) == batch_size or name == files[-1]:
            img_tensor = torch.stack(img).cuda()
            saliency_map = saliency_detect(saliency_model, img_tensor, threshold=30)
            res += (saliency_map.sum(dim=(1, 2)) / (224 * 224)).tolist()
            img.clear()
    
    print(f'Saliency Ratio in {src_dir}: {sum(res) / len(res):.4f}')

if __name__ == '__main__':
    main(src_dir = 'validation/background')
    cal_saliency(src_dir = 'validation/background')