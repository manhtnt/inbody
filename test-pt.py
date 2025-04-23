import torch
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2TokenizerFast,
    LayoutLMv2FeatureExtractor,
    LayoutLMv2ForTokenClassification,
)

# === Load processor và model ===
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")
image_processor = LayoutLMv2FeatureExtractor(apply_ocr=False)
processor = LayoutLMv2Processor(tokenizer=tokenizer, feature_extractor=image_processor)

model = LayoutLMv2ForTokenClassification.from_pretrained("./layoutlmv2-inbody")
model.eval()

# === Load ảnh cần test ===
image_path = "./testing_data/images/0a.jpg"
output_path = "output_infer_a.jpg"

image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"❌ Không đọc được ảnh từ {image_path}")
height, width = image.shape[:2]
image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tesseract config: phân tích theo từng dòng
custom_config = r"--oem 3 --psm 6"
ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
print(ocr_result)
words, boxes = [], []
for i in range(len(ocr_result["text"])):
    word = ocr_result["text"][i].strip()
    if word != "":
        words.append(word)
        x, y, w, h = (
            ocr_result["left"][i],
            ocr_result["top"][i],
            ocr_result["width"][i],
            ocr_result["height"][i],
        )
        boxes.append([x, y, x + w, y + h])


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
print(model.config.id2label)
label_map = model.config.id2label
labels = [label_map[pred] for pred in predictions]
print(predictions)

# === Vẽ kết quả lên ảnh ===
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
word_ids = encoding.word_ids()
used = set()

for idx, word_id in enumerate(word_ids):
    if word_id is None or word_id in used or word_id >= len(boxes):
        continue
    label = labels[idx]
    print((word_id, labels[idx]))
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

# === Lưu ảnh kết quả ===
image.save(output_path)
print(f"✅ Đã lưu ảnh có nhãn vào {output_path}")
image.show()
