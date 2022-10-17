FROM python:3.9.8-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt