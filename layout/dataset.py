import random
from einops import repeat
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image, ImageDraw, ImageOps
import seaborn as sns
import json
import os
from .utils import trim_tokens
from PIL import Image
from tqdm import tqdm


def get_dataset(name, split, data_dir, max_length=None):
    if name == "webui":
        return WebUI(split, data_dir, max_length)
    
    raise NotImplementedError(name)

class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}

class BaseDataset(Dataset):
    component_class = []
    _category_id_to_category_name = None
    _json_category_id_to_contiguous_id = None
    _contiguous_id_to_json_id = None
    _colors = None

    def __init__(self, name, split, is_rela=False):
        super().__init__()
        assert split in ['train', 'val', 'test'], split
        self.is_rela = is_rela
        dir_path = os.path.dirname(os.path.realpath(__file__))
        idx = self.processed_file_names.index('{}.pt'.format(split))
        os.makedirs(os.path.join(dir_path, "preprocess_data", name), exist_ok=True)
        self.data_path = os.path.join(dir_path, "preprocess_data", name, self.processed_file_names[idx])
        self.W = 224 #256
        self.H = 224 #256

    @property
    def json_category_id_to_contiguous_id(self):
        if self._json_category_id_to_contiguous_id is None:
            self._json_category_id_to_contiguous_id = {
            i: i + self.size for i in range(self.categories_num)
        }
        return self._json_category_id_to_contiguous_id

    @property
    def contiguous_category_id_to_json_id(self):
        if self._contiguous_id_to_json_id is None:
            self._contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        return self._contiguous_id_to_json_id

    @property
    def colors(self):
        if self._colors is None:
            num_colors = self.categories_num
            palette = sns.color_palette(None, num_colors)
            if num_colors > 10:
                palette[10:] = sns.color_palette("husl", num_colors-10)
            self._colors = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
        return self._colors

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def quantize_box(self, boxes, width, height):
        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render_normalized_layout(self, layout):
        layout = layout.reshape(-1)
        layout = trim_tokens(torch.tensor(layout), self.bos_token, self.eos_token, self.pad_token).numpy()
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        label = layout[:, 0].astype(np.int32)
        label = label - self.size
        box = box / (self.size - 1)
        return (box, label)

    def render(self, layout, img, border=2, W=224, H=224):
        # img = Image.new('RGB', (self.W, self.H), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(torch.tensor(layout), self.bos_token, self.eos_token, self.pad_token).numpy()
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box = box / (self.size - 1)

        # w, h = img.size
        # box[:, [0, 2]] = box[:, [0, 2]] * (w - 1)
        # box[:, [1, 3]] = box[:, [1, 3]] * (h - 1)
        # box[:, [2, 3]] = box[:, [2, 3]] + 1
        box[:, [0, 2]] = box[:, [0, 2]] * W
        box[:, [1, 3]] = box[:, [1, 3]] * H
        # xywh to ltrb
        x1s = box[:, 0] - box[:, 2] / 2
        y1s = box[:, 1] - box[:, 3] / 2
        x2s = box[:, 0] + box[:, 2] / 2
        y2s = box[:, 1] + box[:, 3] / 2

        for i in range(len(layout)):
            # x1, y1, x2, y2 = box
            x1, y1, x2, y2 = x1s[i], y1s[i], x2s[i], y2s[i]
            cat = layout[i][0]
            assert 0 <= cat-self.size < len(self.colors), 'Invaild Category'
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           )

        # Add border around image
        img = ImageOps.expand(img, border=border)
        return img

    def __getitem__(self, idx):   
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)

        # process image here
        image_path = self.image["image_path"][idx]
        img = Image.open(image_path)
        saliency_path = os.path.join(r'../data/', self.saliency_dir, image_path.split('/')[-1])
        saliency = Image.open(saliency_path)
        saliency = T.functional.to_tensor(saliency)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        
        layout = self.pad(layout.flatten())
        img = T.functional.to_tensor(img)
        saliency = repeat(saliency, 'b w h -> (c b) w h', c=3)
        return layout['x'], layout['y'], img, saliency

    def load_pt(self, load_path):
        results = torch.load(load_path)
        self.categories_num = results["categories_num"]
        self.max_elements_num = results["max_elements_num"]
        self.image = results["image"]
        self.data = results["data"]

        if "iou_data" in results:
            print("load iou data")
            self.iou_data = results["iou_data"]


class WebUI(BaseDataset):
    types2cat = {'static-text': 'text', 'background': 'background', 'image': 'image', 'link-button': 'button', 'button': 'button', 'submit': 'button', 'text': 'text'}
    component_class = {'text': 0, 'background': 1, 'image': 2, 'button': 3}
    def __init__(self, split, data_dir, max_length=None):    
        super().__init__('webui', split)
        self.categories_num = len(self.component_class.keys())
        self.size = 224 # pow(2, precision)
        self.vocab_size = self.size + self.categories_num + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1
        self.max_elements_num = 9
        self.saliency_dir = os.path.join(data_dir, 'saliency')

        if not os.path.exists(self.data_path):
            self.data = []
            self.iou_data = {"bbox":[], "file_idx":[], "file2bboxidx":{}}
            self.image = {"image_path":[], "image_size":[], "canvas_size": []}
            bbox_idx = 0
            meta_dir = os.path.join(data_dir, 'meta')
            img_dir = os.path.join(data_dir, 'layout', 'image')
            dirs = os.listdir(img_dir)         
            for img_name in tqdm(dirs, desc='Processing dataset'):
                website_name = os.path.splitext(img_name)[0]
                meta_path = os.path.join(meta_dir, website_name + '.json')
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                img_path = os.path.join(img_dir, img_name)

                elements = meta['layout']
                if len(elements) == 0 or len(elements) > self.max_elements_num:
                    continue

                ann_box = []
                ann_cat = []
                for ele in elements:
                    if ele['type'] == 'background':
                        img_l, img_t, img_w, img_h = ele['position']
                        break
                
                for ele in elements:
                    if ele['type'] not in self.types2cat:
                        continue       
                    if ele['type'] == 'background':
                        continue
                    
                    l, t, w, h = ele["position"]
                    xc = l + w / 2 - img_l
                    yc = t + h / 2 - img_t
                    if l + w - img_l > img_w or t + h - img_t > img_h:
                        continue
                    if (w * h) / (img_w * img_h) < 15 * 15 / (224 * 224):
                        continue
                    ann_box.append([xc, yc, w, h])
                    ann_cat.append(self.json_category_id_to_contiguous_id[self.component_class[self.types2cat[ele["type"]]]])
                
                if len(ann_box) == 0:
                    continue
                
                # Sort boxes
                ann_box = np.array(ann_box)
                # Discretize boxes
                ann_box = self.quantize_box(ann_box, img_w, img_h)

                ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
                ann_box = ann_box[ind]
                ann_cat = np.array(ann_cat)
                ann_cat = ann_cat[ind]

                self.image["image_path"].append(img_path)
                self.image["image_size"].append((img_w, img_h))
                self.image["canvas_size"].append(meta['size'])

                bbox_idx += 1
                # Append the categories
                layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)
                # ann_cat = np.pad(ann_cat - self.size, (0, self.max_elements_num - len(ann_cat)), 'constant', constant_values=self.categories_num) 

                # Flatten and add to the dataset
                self.data.append((layout.reshape(-1)))
                

            self.save_pt(self.data_path)  

        self.load_pt(self.data_path)
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(item) for item in self.data]) + 2  # bos, eos tokens
        self.pad = Padding(self.max_length, self.vocab_size)

    def save_pt(self, save_path):
        results = {}
        results["categories_num"] = self.categories_num
        results["max_elements_num"] = self.max_elements_num
        results["iou_data"] = self.iou_data
        results["image"] = {}
        
        N = int(len(self.data))
        s = [int(N * .85), int(N * .90)]
        results["data"] = self.data[:s[0]]
        results["image"]["image_path"] = self.image["image_path"][:s[0]]
        results["image"]["image_size"] = self.image["image_size"][:s[0]]
        results["image"]["canvas_size"] = self.image["canvas_size"][:s[0]]
        torch.save(results, os.path.join(os.path.dirname(save_path), self.processed_file_names[0]))
        results["data"] = self.data[s[0]:s[1]]
        results["image"]["image_path"] = self.image["image_path"][s[0]:s[1]]
        results["image"]["image_size"] = self.image["image_size"][s[0]:s[1]]
        results["image"]["canvas_size"] = self.image["canvas_size"][s[0]:s[1]]
        torch.save(results, os.path.join(os.path.dirname(save_path), self.processed_file_names[1]))
        results["data"] = self.data[s[1]:]
        results["image"]["image_path"] = self.image["image_path"][s[1]:]
        results["image"]["image_size"] = self.image["image_size"][s[1]:]
        results["image"]["canvas_size"] = self.image["canvas_size"][s[1]:]
        torch.save(results, os.path.join(os.path.dirname(save_path), self.processed_file_names[2]))
