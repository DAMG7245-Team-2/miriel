FROM python:3.12-slim
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&&apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/backend
COPY --chown=user . $HOME/backend
# CMD ["uvicorn", "api:app", "--port", "8000", "--loop", "asyncio"]