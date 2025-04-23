FROM python:3.10-slim

# Tối ưu tốc độ & tránh lỗi
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    git \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Cài pip trước
RUN pip install --upgrade pip setuptools wheel

# Copy code
WORKDIR /app
COPY . .

# Cài dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "train.py"]
