FROM python:3.11

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy Poetry configuration files
COPY poetry.lock pyproject.toml ./

# Install dependencies (runtime only)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

COPY src/ src/
COPY .env .

EXPOSE 8501
CMD ["streamlit", "run", "src/app/main.py", "--server.fileWatcherType=none", "--server.port=8501", "--server.address=0.0.0.0"]