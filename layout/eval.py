import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch


def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')

def average(scores):
    return sum(scores) / len(scores)

def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = torch.nan_to_num(ai / a1)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.permute(0, 3, 2, 1)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log(1 - X)

    return X.sum(-1) / mask.float().sum(-1)

def main(args):
    # generated layouts
    scores = defaultdict(list)
    for pkl_path in args.pkl_paths:
        alignment, overlap = [], []
        with Path(pkl_path).open('rb') as fb:
            generated_layouts = pickle.load(fb)

        for i in range(0, len(generated_layouts), args.batch_size):
            i_end = min(i + args.batch_size, len(generated_layouts))

            # get batch from data list
            data_list = []
            for b, l in generated_layouts[i:i_end]:
                bbox = torch.tensor(b, dtype=torch.float)
                label = torch.tensor(l, dtype=torch.long)
                data = Data(x=bbox, y=label)
                data_list.append(data)
            data = Batch.from_data_list(data_list)

            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox, _ = to_dense_batch(data.x, data.batch)

            alignment += compute_alignment(bbox, mask).tolist()
            overlap += compute_overlap(bbox, mask).tolist()

        alignment = average(alignment)
        overlap = average(overlap)

        scores['Alignment'].append(alignment)
        scores['Overlap'].append(overlap)

    print(f'Dataset: {args.dataset}')
    print_scores(scores)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='dataset name',
                        choices=['rico', 'publaynet', 'webui'])
    parser.add_argument('pkl_paths', type=str, nargs='+',
                        help='generated pickle path')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='input batch size')
    parser.add_argument('--compute_real', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    main(args)