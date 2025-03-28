# Start from an official Airflow image
FROM apache/airflow:2.8.0-python3.11

# Switch to root to install system dependencies
USER root

# Install Rust (latest stable) and build essentials
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    git \
    pkg-config \
    libssl-dev \
    && curl https://sh.rustup.rs -sSf | bash -s -- -y \
    && apt-get clean

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Switch back to airflow user
USER airflow

# Install Python requirements
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt
