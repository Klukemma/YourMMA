FROM python:3.12-slim

# Toolchains the agents use. Agents themselves cannot apt-get (no root).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential nodejs npm zip unzip jq procps \
 && rm -rf /var/lib/apt/lists/*

# Agents' commands run as this unprivileged user. The server itself runs as
# root inside the container only so it can drop to this user per command and
# keep the database and API keys out of the agents' reach.
RUN useradd --create-home --uid 1001 --shell /bin/bash agent \
 && su agent -c "git config --global user.name 'AI Team' && git config --global user.email 'ai-team@users.noreply.github.com' && git config --global init.defaultBranch main"

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY agents.yaml .

ENV DATA_DIR=/data PYTHONUNBUFFERED=1
RUN mkdir -p /data && chmod 700 /data
VOLUME ["/data"]
EXPOSE 8000
CMD ["sh", "-c", "chmod 711 /data && mkdir -p /data/workspaces && chmod 711 /data/workspaces && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
