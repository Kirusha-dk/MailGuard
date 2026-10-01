FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir numpy>=2.0,<3 \
 && python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ml/ /app/

CMD ["python", "train.py", "--help"]
