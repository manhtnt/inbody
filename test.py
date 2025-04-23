import torch
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from paddleocr import PaddleOCR
import numpy as np

from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2TokenizerFast,  # ✅ dùng phiên bản Fast
    LayoutLMv2FeatureExtractor,
    LayoutLMv2ForTokenClassification,
)

tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
image_processor = LayoutLMv2FeatureExtractor(apply_ocr=False)
processor = LayoutLMv2Processor(tokenizer=tokenizer, feature_extractor=image_processor)

# === Load model đã fine-tune ===
model = LayoutLMv2ForTokenClassification.from_pretrained("./layoutlmv2-inbody")
model.eval()

# === Load ảnh cần test ===
image_path = "./testing_data/images/a.jpg"
output_path = "output_infer_a.jpg"
image = Image.open(image_path).convert("RGB")
width, height = image.size

# === OCR với pytesseract ===
# ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
# words, boxes = [], []

# for i in range(len(ocr_result["text"])):
#     word = ocr_result["text"][i].strip()
#     if word != "":
#         words.append(word)
#         x, y, w, h = (
#             ocr_result["left"][i],
#             ocr_result["top"][i],
#             ocr_result["width"][i],
#             ocr_result["height"][i],
#         )
#         boxes.append([x, y, x + w, y + h])

ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

# === OCR với PaddleOCR ===
results = ocr_engine.ocr(image_path, cls=True)

words, boxes = [], []

for line in results[0]:
    box = line[0]  # list of 4 points (x,y)
    text = line[1][0].strip()
    if not text:
        continue

    x1 = min([point[0] for point in box])
    y1 = min([point[1] for point in box])
    x2 = max([point[0] for point in box])
    y2 = max([point[1] for point in box])

    words.append(text)
    boxes.append([x1, y1, x2, y2])


# === Chuẩn hóa box về [0, 1000] ===
def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


norm_boxes = [normalize_box(b, width, height) for b in boxes]

# === Encode input ===
encoding = processor(
    image,
    text=words,
    boxes=norm_boxes,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512,
)

# === Dự đoán ===
with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()

# === Decode label
label_map = model.config.id2label
labels = [label_map[pred] for pred in predictions]

# === Hiển thị nhãn lên ảnh ===
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

word_ids = encoding.word_ids()
used = set()

for idx, word_id in enumerate(word_ids):
    if word_id is None or word_id in used or word_id >= len(boxes):
        continue

    label = labels[idx]
    if label != "O":
        box = boxes[word_id]
        draw.rectangle(box, outline="red", width=2)
        draw.text(
            (box[0] - 20, box[1] - 20),
            label.split("-")[1][0] + label.split("-")[2],
            fill="blue",
            font=font,
        )
    used.add(word_id)

# === Lưu kết quả ===
image.save(output_path)
print("✅ Kết quả đã lưu vào output_infer.jpg")
image.show()
