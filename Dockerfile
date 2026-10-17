FROM python:3.12-slim
RUN addgroup --system --gid 10001 ragcheckup \
    && adduser --system --uid 10001 --ingroup ragcheckup ragcheckup
WORKDIR /app
COPY --chown=ragcheckup:ragcheckup . /app
RUN pip install --no-cache-dir . && rm -rf /app/.venv \
    && chown -R ragcheckup:ragcheckup /app
USER ragcheckup
ENTRYPOINT ["rag-checkup"]
# 默认零 key 演示:docker run --rm rag-checkup demo
CMD ["demo"]
