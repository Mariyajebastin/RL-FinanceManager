# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest

FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

# git is needed for VCS installs; curl is used only if the base image lacks uv.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt uv.lock ./

# Ensure uv is available even when building outside the OpenEnv base image.
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install the environment runtime from pyproject plus inference extras from requirements.txt.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi && \
    uv pip install --python .venv/bin/python -r requirements.txt

COPY . .

FROM ${BASE_IMAGE}

WORKDIR /app/env

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/env:${PYTHONPATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Use Python for the healthcheck so the image does not depend on curl at runtime.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
