import os
import requests
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

SAVE_DIR = f'data/raw'
os.makedirs(SAVE_DIR, exist_ok=True)


def download_image(url, path):
    res = requests.get(url)
    if res.status_code == 200:
        with open(path, 'wb') as f:
            f.write(res.content)


def extract_image_url(url):
    download_url = url[5: -2] if url.startswith('url') else url
    return download_url if download_url.startswith('http') else None


def process_data(data):
    website = data[1]['website']
    url = extract_image_url(data[1]['url'])
    save_path = os.path.join(SAVE_DIR, f'{website}.jpg')
    
    try:
        if url is not None:
            download_image(url, save_path)
        
    except Exception as e:
        print(website, str(e))


if __name__ == '__main__':
    data = pd.read_csv('data/hash2url.csv')
    with Pool(1) as workers:
        with tqdm(total=len(data)) as pbar:
            for i in workers.imap(process_data, data.iterrows()):
                pbar.update()
