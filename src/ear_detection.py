import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

import cv2
import torch
import time
from pathlib import Path
import numpy as np

# ====================== CONFIG ======================
INPUT_DIR     = r'path\ear\image'     # folder with input images
OUTPUT_DIR    = r'path\ear\crops'     # where to save cropped ears
YOLOV5_REPO   = r'path\yolov5'        # local yolov5 repo
MODEL_PATH    = r'path\model.pt' # your ear model

CONF_THRES    = 0.5
IOU_THRES     = 0.4
SAVE_ALL_BOXES = False   # set True to save every detected box per image
# ====================================================

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# Load model
model = torch.hub.load(YOLOV5_REPO, 'custom', path=MODEL_PATH, source='local')
model.to(device)
model.conf = CONF_THRES
model.iou  = IOU_THRES

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

image_paths = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in IMG_EXTS]
print(f"🔎 Found {len(image_paths)} images.")

num_saved = 0
num_processed = 0
t0 = time.time()

for img_path in image_paths:
    num_processed += 1
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read image: {img_path.name}")
        continue

    # Run detection
    result = model(img)
    preds = result.xyxy[0]  # tensor: (x1, y1, x2, y2, conf, cls)
    if preds is None or len(preds) == 0:
        print(f"⚠️ No ear detected in: {img_path.name}")
        continue

    preds = preds.cpu().numpy()

    if SAVE_ALL_BOXES:
        selected = preds
    else:
        selected = preds[[preds[:, 4].argmax()]]  # best (highest conf) only

    for j, (x1, y1, x2, y2, conf, cls) in enumerate(selected):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ Empty crop for {img_path.name} (box idx {j}) – skipping.")
            continue

        if SAVE_ALL_BOXES:
            out_name = f"{img_path.stem}_crop_{j:02d}_{int(conf*100)}.jpg"
        else:
            out_name = f"{img_path.stem}_crop_{int(conf*100)}.jpg"

        out_path = Path(OUTPUT_DIR) / out_name
        cv2.imwrite(str(out_path), crop)
        num_saved += 1

    print(f"✅ {img_path.name}: saved {len(selected)} crop(s).")

print(f"\n🎯 Done. Processed: {num_processed} images, saved: {num_saved} crops in {OUTPUT_DIR}")
print(f"⏱️ Elapsed: {time.time() - t0:.2f}s")
