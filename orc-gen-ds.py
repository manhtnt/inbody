import cv2
import json
import os
from paddleocr import PaddleOCR

# from transformers import AutoModel
# model = AutoModel.from_pretrained("microsoft/layoutlmv2-base-uncased")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
)  #

image_paths = [f"{i}.jpg" for i in range(1, 21)]
image_dir = "./training_data/images/"
annotation_dir = "./training_data/annotations/"


def extract_ocr_data(image_path):
    results = ocr.ocr(image_path, cls=True)
    lines = []
    for id, line in enumerate(results[0][:15]):
        ((x1, y1), _, (x2, y2), _) = line[0]
        text = line[1][0]
        print(id, text, line[0])
        lines.append(
            {
                "box": [x1, y1, x2, y2],
                "text": text,
                # "words": [
                #     {
                #         "text": text,
                #         "box": [x1, y1, x2, y2],
                #     }
                # ],
                "label": "B-KEY-ID",
            }
        )
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
    full_image_path = os.path.join(image_dir, image_path)
    print(f"🔍 Đang xử lý ảnh: {image_path}")
    img = cv2.imread(full_image_path)
    if img is None:
        print(f"⚠️ Không tìm thấy ảnh: {image_path}")
        continue

    h, w = img.shape[:2]
    ocr_lines = extract_ocr_data(full_image_path)
    # print(ocr_lines)
    # print("===" * 20)
    # save to json file
    with open(
        os.path.join(annotation_dir, f"{image_path.split('.')[0]}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ocr_lines, f, ensure_ascii=False, indent=4)
    # matched_data = match_key_value(ocr_lines, w)

# Xuất kết quả
# print(json.dumps(final_results, indent=4, ensure_ascii=False))
