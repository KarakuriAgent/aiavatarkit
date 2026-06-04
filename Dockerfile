FROM python:3.12-slim

ARG TARGETARCH
ARG TARGETVARIANT
ARG DEEPFILTER_VERSION=0.5.6

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.7.3

RUN set -eux; \
    case "${TARGETARCH}${TARGETVARIANT:-}" in \
      "arm64") deepfilter_target="aarch64-unknown-linux-gnu" ;; \
      "amd64") deepfilter_target="x86_64-unknown-linux-musl" ;; \
      "armv7") deepfilter_target="armv7-unknown-linux-gnueabihf" ;; \
      *) echo "Unsupported deep-filter target: ${TARGETARCH}${TARGETVARIANT:-}" >&2; exit 1 ;; \
    esac; \
    curl -L --fail \
      "https://github.com/Rikorose/DeepFilterNet/releases/download/v${DEEPFILTER_VERSION}/deep-filter-${DEEPFILTER_VERSION}-${deepfilter_target}" \
      -o /usr/local/bin/deep-filter; \
    chmod +x /usr/local/bin/deep-filter; \
    /usr/local/bin/deep-filter --version

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uv", "run", "server"]
