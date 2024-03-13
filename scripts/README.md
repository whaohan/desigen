# WebUI data processing

## Usage

1. Download all images.

   ```python
   python3 download_img.py
   ```

2. Resize the background images.

   ```python
   python3 resize.py
   ```

3. Process the saliency map.

   ```python
   python3 saliency.py
   ```

4. Copy background images from `data/image` directory to `data/background` and `data/layout` based on `background*.json` and `layout.json`