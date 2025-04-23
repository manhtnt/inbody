import cv2
import json
from paddleocr import PaddleOCR


ocr = PaddleOCR(use_angle_cls=True, lang="en")

image_paths = ["s2.jpg"]


def split_box_by_text(box, text):
    """Chia box lớn thành các box nhỏ ứng với từng từ trong text."""
    tokens = text.strip().split()
    if len(tokens) <= 1:
        return [(text, box)]

    x1, y1, x2, y2 = box
    total_width = x2 - x1
    step = total_width / len(tokens)

    results = []
    for i, token in enumerate(tokens):
        token_x1 = int(x1 + i * step)
        token_x2 = int(x1 + (i + 1) * step)
        token_box = [token_x1, y1, token_x2, y2]
        results.append((token, token_box))
    return results


def extract_ocr_data(image_path):
    results = ocr.ocr(image_path, cls=True)
    lines = []
    id_counter = 0

    for line in results[0]:
        ((x1, y1), _, (x2, y2), _) = line[0]
        full_text = line[1][0]

        for token_text, token_box in split_box_by_text([x1, y1, x2, y2], full_text):
            lines.append(
                {
                    "id": id_counter,
                    "box": token_box,
                    "text": token_text,
                    "words": [
                        {
                            "text": token_text,
                            "box": token_box,
                        }
                    ],
                    "x": (token_box[0] + token_box[2]) / 2,
                    "y": (token_box[1] + token_box[3]) / 2,
                }
            )
            id_counter += 1
    return lines


def match_key_value(lines, img_width, threshold=25):
    left = [l for l in lines if l["x"] < img_width * 0.5]
    right = [l for l in lines if l["x"] >= img_width * 0.5]

    left = sorted(left, key=lambda l: l["y"])
    right = sorted(right, key=lambda l: l["y"])

    result = {}
    for key in left:
        for value in right:
            if abs(key["y"] - value["y"]) < threshold:
                result[key["text"]] = value["text"]
                break
    return result


final_results = []
for image_path in image_paths:
    print(f"🔍 Đang xử lý ảnh: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Không tìm thấy ảnh: {image_path}")
        continue

    h, w = img.shape[:2]
    ocr_lines = extract_ocr_data(image_path)

    # Xuất ra file JSON
    with open(f"{image_path.split('.')[0]}_split.json", "w", encoding="utf-8") as f:
        json.dump(ocr_lines, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu OCR tách box tại: {image_path.split('.')[0]}_split.json")
