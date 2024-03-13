import os
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from .saliency import get_saliency_model, saliency_detect

if __name__ == '__main__':
    BASE = r'data/image'
    SALIENCY = r'data/saliency/'
    os.makedirs(SALIENCY, exist_ok=True)
    img_paths = os.listdir(BASE)
    model = get_saliency_model()
    batch_size = 128
    img_list = []
    name_list = []
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            base_path = os.path.join(BASE, img_path)
            img = torchvision.transforms.functional.to_tensor(Image.open(base_path)).unsqueeze(0)
            img_list.append(img)
            name_list.append(img_path)
            if len(img_list) == batch_size or img_path == img_paths[-1]:
                inp = torch.cat(img_list, dim=0).cuda()
                smap = saliency_detect(model, inp, threshold=None)
                for i in range(len(smap)):
                    smap_img = torchvision.transforms.functional.to_pil_image(smap[i].unsqueeze(0))
                    smap_img.save(os.path.join(SALIENCY, name_list[i]))
                img_list, name_list = [], []