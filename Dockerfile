FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System deps (optional but helpful for scientific stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy app code
COPY app/ app/
COPY students/ students/
COPY github.txt pyproject.toml ./

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.fileWatcherType=none", "--server.port=8501", "--server.address=0.0.0.0"]