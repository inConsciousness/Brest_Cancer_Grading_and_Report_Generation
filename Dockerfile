# ✅ Use official lightweight Python 3.10 image
FROM python:3.10-slim

# ✅ Set working directory inside the container
WORKDIR /app

# ✅ Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ✅ Set TensorFlow/Keras environment variables
ENV KERAS_BACKEND=tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=2

# ✅ Copy and install Python dependencies (Keras 3.x compatible)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ Copy the entire FastAPI project
COPY . .

# ✅ Expose FastAPI app port
EXPOSE 8000

# ✅ Start the FastAPI server (main app lives at app/main.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
