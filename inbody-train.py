import os
import json
from datasets import Dataset
from transformers import (
    LayoutLMv2Processor,
    LayoutLMv2ForTokenClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from PIL import Image
import numpy as np

# === Label mapping ===
label_list = [
    "O",
    "B-KEY-ID",
    "B-VALUE-ID",
    "B-KEY-HEIGHT",
    "B-VALUE-HEIGHT",
    "B-KEY-AGE",
    "B-VALUE-AGE",
    "B-KEY-GENDER",
    "B-VALUE-GENDER",
    "B-VALUE-TIME",
]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

# === Load processor (no OCR) ===
processor = LayoutLMv2Processor.from_pretrained(
    "microsoft/layoutlmv2-base-uncased", apply_ocr=False
)


# === Load and prepare data ===
def load_data(json_folder, image_folder):
    samples = []
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), encoding="utf-8") as f:
                data = json.load(f)
            words = [item["text"] for item in data]
            boxes = [[int(coord) for coord in item["box"]] for item in data]
            labels = [item["label"] for item in data]

            samples.append(
                {
                    "id": file.split(".")[0],
                    "words": words,
                    "bboxes": boxes,
                    "labels": labels,
                    "image_path": os.path.join(
                        image_folder, file.replace(".json", ".jpg")
                    ),
                }
            )
    return samples


# === Preprocess ===
import numpy as np


def normalize_box(box, width, height):
    return [
        int(1000 * box[0] / width),
        int(1000 * box[1] / height),
        int(1000 * box[2] / width),
        int(1000 * box[3] / height),
    ]


def preprocess(example):
    image = Image.open(example["image_path"]).convert("RGB")
    width, height = image.size  # Lấy size trước khi chuyển sang numpy
    boxes = [normalize_box(box, width, height) for box in example["bboxes"]]

    image = np.array(image).astype(np.uint8)  # convert sau

    encoding = processor(
        image,
        text=example["words"],
        boxes=boxes,
        word_labels=[label2id[label] for label in example["labels"]],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    encoding = {k: v.squeeze(0) for k, v in encoding.items()}
    return encoding


# === Load data ===
data_dir = "./training_data"
examples = load_data(
    json_folder=os.path.join(data_dir, "annotations"),
    image_folder=os.path.join(data_dir, "images"),
)
dataset = Dataset.from_list(examples)
dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# === Model ===
model = LayoutLMv2ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv2-base-uncased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# === Trainer ===
training_args = TrainingArguments(
    output_dir="./layoutlmv2-inbody",
    per_device_train_batch_size=2,
    num_train_epochs=100,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="no",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=default_data_collator,
    tokenizer=processor,
)

trainer.train()
trainer.save_model("./layoutlmv2-inbody")
