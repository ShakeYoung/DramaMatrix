# DramaMatrix 容器镜像（W7）：Python 3.11 + ffmpeg，开箱可跑流水线。
# 构建：docker build -t dramamatrix .
# 运行：
#   docker run --rm -v "$PWD/data:/app/data" --env-file code/.env dramamatrix \
#     python main.py --project-id demo
# 说明：
# - /app/code/outputs 与 SQLite 库写入容器内；如需持久化请挂载卷。
# - .env 通过 --env-file 注入，绝不打入镜像。

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/code

COPY pyproject.toml README.md requirements.txt ./
COPY code ./code

RUN pip install --no-cache-dir .

ENV DRAMAMATRIX_OUTPUT_DIR=/app/data/outputs \
    PYTHONUNBUFFERED=1

# 默认入口即流水线 CLI；看板/审阅台用 `docker run ... python -m src.dashboard_server` 覆盖。
WORKDIR /app/code/code
ENTRYPOINT ["python", "main.py"]
