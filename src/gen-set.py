import cv2
import json
import os
import pytesseract
from PIL import Image

image_paths = [f"{i}.jpg" for i in range(1, 21)]
image_dir = "./training_data/images/"
annotation_dir = "./training_data/annotations/"


def extract_ocr_data(image_path):
    image = Image.open(image_path).convert("RGB")
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    lines = []
    for id in range(min(20, len(ocr_result["text"]))):
        text = ocr_result["text"][id].strip()
        if text == "":
            continue
        x, y, w, h = (
            ocr_result["left"][id],
            ocr_result["top"][id],
            ocr_result["width"][id],
            ocr_result["height"][id],
        )
        x1, y1, x2, y2 = x, y, x + w, y + h
        print(id, text, [x1, y1, x2, y2])
        lines.append(
            {
                "box": [x1, y1, x2, y2],
                "text": text,
                "label": "B-KEY-ID",
            }
        )
    return lines


# Xử lý từng ảnh
for image_path in image_paths:
    full_image_path = os.path.join(image_dir, image_path)
    print(f"🔍 Đang xử lý ảnh: {image_path}")
    img = cv2.imread(full_image_path)
    if img is None:
        print(f"⚠️ Không tìm thấy ảnh: {image_path}")
        continue

    ocr_lines = extract_ocr_data(full_image_path)

    # Ghi file JSON
    output_path = os.path.join(annotation_dir, f"{image_path.split('.')[0]}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_lines, f, ensure_ascii=False, indent=4)
