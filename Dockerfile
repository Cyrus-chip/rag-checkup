FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir . && rm -rf /app/.venv
ENTRYPOINT ["rag-checkup"]
# 默认零 key 演示:docker run --rm rag-checkup demo
CMD ["demo"]
