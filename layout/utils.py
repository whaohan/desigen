import random
import numpy as np
import torch
from torch.nn import functional as F
import seaborn as sns
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(adj):
    B, N, _ = adj.shape
    adj_ = adj + torch.eye(N, device=adj.device)
    rowsum = adj_.sum(dim=2)
    mat1 = torch.pow(rowsum, -0.5).reshape(B, N, 1).repeat(1, 1, N) * torch.eye(N, device=adj.device)
    mat2 = torch.pow(rowsum, -0.5).reshape(B, N, 1).repeat(1, 1, N) * torch.eye(N, device=adj.device)
    adj_norm = mat1.bmm(adj_).bmm(mat2)
    return adj_norm

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette(None, num_colors)
    if num_colors > 10:
        palette[10:] = sns.color_palette("husl", num_colors-10)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


@torch.no_grad()
def sample(model, x, img_feature, saliency, steps, temperature=1.0, sample=False, top_k=None, only_label=False, gt=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.module.get_block_size() if hasattr(model, "module") else model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        if only_label == True and k % 5 == 0:
            ix = gt[:, k+1].unsqueeze(1)
        else:
            logits, _ = model(img_feature, x_cond, saliency=None)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    return x


def trim_tokens(tokens, bos, eos, pad=None):
    bos_idx = torch.where(tokens == bos)[0]
    tokens = tokens[bos_idx[0]+1:] if len(bos_idx) > 0 else tokens
    eos_idx = torch.where(tokens == eos)[0]
    tokens = tokens[:eos_idx[0]] if len(eos_idx) > 0 else tokens
    # tokens = tokens[tokens != bos]
    # tokens = tokens[tokens != eos]
    if pad is not None:
        tokens = tokens[tokens != pad]
    return tokens

def seq_to_bbox(layout_seq, bos, eos, pad):
    layout = trim_tokens(torch.tensor(layout_seq), bos, eos, pad).numpy()
    layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
    box = layout[:, 1:].astype(np.float32)
    cat = layout[:, 0]
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    return cat, box.astype(np.int32)

def layout_to_mask(layout, bos, eos, pad):
    _, bboxes = seq_to_bbox(layout, bos, eos, pad)
    mask = np.zeros((224, 224))
    for box in bboxes:
        l, t, w, h = box
        mask[t: t+h, l: l+w] = 1
    return mask

def random_mask(adj, mask_ratio, dire_len=9):
    # sample mask randomly from [0, 1)
    mask = torch.rand(adj.shape, device=adj.device) < mask_ratio
    adj[mask] = dire_len + 1
    return adj

def random_noise(adj, noise_ratio, dire_len):
    # sample mask randomly from [0, 1)
    noise = torch.rand(adj.shape, device=adj.device)
    
    mask_plus = noise < noise_ratio / 2
    adj[mask_plus] = (adj[mask_plus] + 1) % dire_len
    mask_minus = noise > 1 - noise_ratio / 2
    adj[mask_minus] = (adj[mask_minus] + dire_len - 1) % dire_len

    return adj

def get_mask(w, h, factor=0.2):
    img = Image.new('RGB', (w, h), 'white')
    img = img.convert('RGBA')
    img_blender = Image.new('RGBA', img.size, (0, 0, 0, 0))
    img = Image.blend(img, img_blender, factor)
    img_temp = img
    return img_temp


def add_mask(base_img, region):
    for re in region:
        l, t, w, h = re
        mask = get_mask(w, h)
        a = mask.split()[3]
        base_img.paste(mask, (l, t, l+w, t+h), a)
    return base_img


def combine_regions(regions, margin_l=5, margin_t=10):
    l = max(regions[:, 0].min() - margin_l, 0)
    t = max(regions[:, 1].min() - margin_t, 0)
    r = min((regions[:, 0] + regions[:, 2]).max() + margin_l, 223)
    b = min((regions[:, 1] + regions[:, 3]).max() + margin_t, 223)
    return (l, t, r-l, b-t)