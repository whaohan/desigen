from PIL import Image
import torchvision.transforms as T
import os
import json
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool


data_path = r'data'
source_img_dir = os.path.join(data_path, 'raw')
target_img_dir = os.path.join(data_path, 'image')
meta_dir = os.path.join(data_path, 'meta')
os.makedirs(target_img_dir, exist_ok=True)
H, W = 512, 512


def get_image_size(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    for ele in meta['layout']:
        if ele['type'] == 'background':
            _, _, img_w, img_h = ele['position']
            return img_w, img_h
        
    return None

def resize_image(website):
    img_path = os.path.join(source_img_dir, website)
    website_name = os.path.splitext(website)[0]
    meta_path = os.path.join(meta_dir, website_name + '.json')
    img_w, img_h = get_image_size(meta_path)
    
    img = Image.open(img_path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    # two modes for resizing: based on width / based on height
    if int(img_w / img.width * img.height) >= img_h:
        img = T.functional.resize(img, (int(img_w / img.width * img.height), img_w))
    else:
        img = T.functional.resize(img, (img_h, int(img_h / img.height * img.width)))
    img = T.functional.center_crop(img, (img_h, img_w))
    img = T.functional.resize(img, (H, W))
    img.save(os.path.join(target_img_dir, website))
    

if __name__ == '__main__':
    websites = os.listdir(source_img_dir)
    with Pool(64) as workers:
        with tqdm(total=len(websites)) as pbar:
            for i in workers.imap(resize_image, websites):
                pbar.update()