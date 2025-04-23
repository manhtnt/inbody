import torch
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3TokenizerFast,
)

model_name = "nielsr/layoutlmv3-finetuned-funsd"
tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name)

# Tắt apply_ocr để tự cung cấp words và boxes
image_processor = LayoutLMv3ImageProcessor(apply_ocr=False)

# Tạo processor hợp lệ
processor = LayoutLMv3Processor(
    tokenizer=tokenizer, image_processor=image_processor, apply_ocr=False
)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
model.eval()

# === Load ảnh ===
image = Image.open("03.jpg").convert("RGB")
width, height = image.size

# === OCR (dùng pytesseract) ===
ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
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


# === Normalize boxes về [0, 1000] ===
def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


norm_boxes = [normalize_box(b, width, height) for b in boxes]

# === Encode input (không dùng apply_ocr) ===
encoding = processor(
    images=image,
    text=words,  # <-- PHẢI là `text`, không phải `words`
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

# === Giải mã nhãn ===
label_map = model.config.id2label
labels = [label_map[p] for p in predictions]

# === Vẽ kết quả lên ảnh ===
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

word_ids = encoding.word_ids()
labels = [label_map[p] for p in predictions]

used = set()
for idx, word_id in enumerate(word_ids):
    if word_id is None or word_id in used:
        continue

    label = labels[idx]
    if label != "O":
        word = words[word_id]
        box = boxes[word_id]

        # Vẽ khung đỏ quanh box
        # draw.rectangle(box, outline="red", width=2)

        # Viết nhãn lên trên box
        draw.text((box[0] + 2, box[1] - 10), label[2:5], fill="red", font=font)

        # In thông tin
        print(f"{word} → {label}")

    used.add(word_id)

# === Lưu ảnh kết quả ===
image.save("output_lmv3_funsd_labeled.png")
print("✅ Đã lưu ảnh gán nhãn thành output_lmv3_funsd_labeled.png")
image.show()
