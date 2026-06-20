FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Demo dependencies only (lightweight, no PyTorch needed)
COPY src/Demo/requirements.txt /app/requirements-demo.txt
RUN pip install --no-cache-dir -r /app/requirements-demo.txt

# Copy demo source
COPY src/Demo/app.py src/Demo/inference.py src/Demo/telemetry.py /app/src/Demo/
COPY src/Demo/static/ /app/src/Demo/static/

# Model weights -- user must provide ONNX file
# docker cp your_model.onnx container:/app/src/Demo/models/
RUN mkdir -p /app/src/Demo/models

EXPOSE 8000

CMD ["uvicorn", "src.Demo.app:app", "--host", "0.0.0.0", "--port", "8000"]
