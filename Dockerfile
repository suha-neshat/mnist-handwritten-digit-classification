FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \ 
    libjpeg62-turbo-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/
COPY model/mnist_cnn.py /app/
# NOTE: Ensure mnist_cnn.pt exists before building the image (run training first)
COPY model/mnist_cnn.pt /app/

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]