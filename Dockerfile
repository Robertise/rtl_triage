# ==============================================================================
# RTL Failure Triage Agent - Simulation Environment Dockerfile
# ==============================================================================
# Fixed base: Ubuntu 24.04 LTS (Noble Numbat)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies: Icarus Verilog 12, Python 3.12, venv, build tools, libpython dev, procps
RUN apt-get update && apt-get install -y --no-install-recommends \
    iverilog=12.0-2build2 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Create virtual environment and install pinned requirements
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "scripts/run_sim.py", "--help"]
