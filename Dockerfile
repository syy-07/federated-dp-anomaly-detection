FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "src/train_fed.py", "--config", "configs/fed_dp_small.yml"]
