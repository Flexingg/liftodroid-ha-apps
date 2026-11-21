ARG BUILD_FROM
FROM ${BUILD_FROM}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps "rapidocr_onnxruntime>=1.3,<2"

COPY barbell /app/barbell
COPY barbell.example.yaml /app/barbell.example.yaml
WORKDIR /app
CMD ["/opt/venv/bin/python", "-m", "barbell"]
