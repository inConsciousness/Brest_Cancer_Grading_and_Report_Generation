# ✅ Use official lightweight Python image
FROM python:3.10-slim

# ✅ Set working directory inside the container
WORKDIR /app

# ✅ Install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ✅ Set environment to suppress TensorFlow logs
ENV KERAS_BACKEND=tensorflow
ENV TF_CPP_MIN_LOG_LEVEL=2

# ✅ Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ✅ Copy the entire project into the container
COPY . .

# ✅ Expose port for FastAPI
EXPOSE 8000

# ✅ Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
