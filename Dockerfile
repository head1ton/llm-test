FROM python:3.11-slim

WORKDIR /app

# 기본 유틸
RUN apt-get update && apt-get install -y --no-install-recommands \
    curl \
    && rm -rf /var/lib/apt/lists/*

# UV 설치
RUN pip install --no-cache-dir uv

# 프로젝트 복사
COPY . /app

# 의존성 설치 (uv.lock / pyproject.toml 기준)
# uv 사용 방식에 따라 둘 중 하나:
# 1) uv sync (권장: lock 기반)
RUN uv sync --frozen

# 2) 또는 requirements 기반이면
# RUN uv pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# stdio MCP인 경우 workers=1 권장
CMD ["uv", "run", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]