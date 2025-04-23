import os
import cv2
import json
import pytesseract
from pytesseract import Output
import re


# === Cấu hình đường dẫn ===
image_folder = "training_data/img"
# output_folder = "training_data/annotations"
output_folder = "training_data/ocr"
os.makedirs(output_folder, exist_ok=True)


# === Hàm gán nhãn dựa vào từ
def get_label_for_text(text):
    text = text.lower()

    if text == "height":
        return "B-KEY-HEIGHT"
    elif text == "gender":
        return "B-KEY-GENDER"
    elif text == "age":
        return "B-KEY-AGE"
    elif text == "id":
        return "B-KEY-ID"
    elif text == "test":
        return "B-KEY-TIME"
    elif text == "date":
        return "I-KEY-TIME"
    elif text == "time":
        return "E-KEY-TIME"
    elif text in ("male", "female"):
        return "B-VALUE-GENDER"
    elif "cm" in text:
        return "B-VALUE-HEIGHT"
    elif re.fullmatch(r"\d{10}", text):
        return "B-VALUE-ID"
    else:
        return "O"


def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


# === Hàm OCR cho từng ảnh (không chia box)
def extract_ocr_data(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tăng tương phản + tách nền
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract config: phân tích theo từng dòng
    custom_config = r"--oem 3 --psm 6"

    data = pytesseract.image_to_data(thresh, config=custom_config, output_type=Output.DICT)

    height, width = image.shape[:2]
    lines = []
    id_counter = 0
    n = len(data["text"])

    for i in range(n):
        word = data["text"][i].strip()
        level = int(data["level"][i])

        # Loại bỏ box không phải từ hoặc từ rỗng
        if word == "" or level != 5:
            continue

        # Lấy độ tin cậy nếu có
        try:
            conf = int(float(data["conf"][i]))
        except:
            conf = -1

        if conf < 0:
            continue

        # Box pixel gốc
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        box = [x, y, x + w, y + h]
        norm_box = normalize_box(box, width, height)

        label = get_label_for_text(word)

        lines.append({
            "id": id_counter,
            "box": box,
            "norm_box": norm_box,
            "text": word,
            "label": label,
            "conf": conf,
        })
        id_counter += 1

    return lines


# === Chạy toàn bộ ảnh ===
# === Chạy toàn bộ ảnh ===
os.makedirs("image-ocr", exist_ok=True)

image_files = [
    f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))
]
for img_file in sorted(image_files):
    img_path = os.path.join(image_folder, img_file)
    ocr_lines = extract_ocr_data(img_path)

    # === Lưu annotation JSON
    out_file = os.path.join(output_folder, img_file.rsplit(".", 1)[0] + ".json")
    result = list(
        map(
            lambda line: {
                "id": line["id"],
                "box": line["norm_box"],
                "text": line["text"],
                "label": line["label"],
            },
            ocr_lines,
        )
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu tại: {out_file}")

    # === Vẽ box và text lên ảnh
    image = cv2.imread(img_path)  # chỉ đọc 1 lần duy nhất
    for line in ocr_lines:
        x1, y1, x2, y2 = line["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            line["text"] + "_" + str(line["conf"]),
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            1,
        )

    # === Lưu ảnh đã annotate
    out_img_path = os.path.join("training_data/image-ocr", img_file)
    cv2.imwrite(out_img_path, image)
    print(f"🖼️  Đã lưu ảnh có OCR box: {out_img_path}")
