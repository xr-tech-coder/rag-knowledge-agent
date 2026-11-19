# 为什么用 3.12 而不是本地的 3.14：numpy 2.x 需要 ≥3.11，3.12 是当前生态最稳的一档，
# 3.14 太新，slim 镜像里的 wheel 覆盖还不全。
FROM python:3.12-slim

# 不生成 .pyc、日志不打缓冲（不然 docker logs 看不到流式输出）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 依赖单独一层：改代码不会触发重装，rebuild 只要几秒。
# requirements.txt 里只有四条轻依赖，不含 sentence-transformers ——
# 后者会连带装 2GB 的 torch，而线上默认 RERANK=0 根本用不到。
# 真要开重排，把 requirements-rerank.txt 也 COPY 进来装一次即可。
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

# streamlit 自带的健康检查端点。/_stcore/health 返回 ok 才算起好了。
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/_stcore/health', timeout=4).read()==b'ok' else 1)"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
