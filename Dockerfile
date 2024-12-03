ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}
ARG POETRY_VERSION=1.8.2
RUN pip install poetry==${POETRY_VERSION}

WORKDIR /app
COPY ./ ./
RUN poetry config virtualenvs.in-project true \
    && poetry install --only main --no-root \
    && . .venv/bin/activate \
    && pip install --no-deps .
