FROM artefact.skao.int/ska-build-python-ubuntu24:1.1.2 AS builder

ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=1
ENV POETRY_VIRTUALENVS_CREATE=1

# Use /app so the virtual environment path matches the runner
WORKDIR /app

COPY pyproject.toml poetry.lock README.md LICENSE ./

# Install dependencies first to cache the layer
RUN poetry install --only main --no-root

# Install the package itself (non-editable) into the virtual environment
COPY src ./src
COPY scripts/clean-with-gains scripts/oskar_csv_converter.py scripts/gleam_csv_converter.py ./scripts/
RUN .venv/bin/pip install --no-cache-dir --no-dependencies .


FROM artefact.skao.int/ska-python-ubuntu24:1.1.2 AS runner

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TZ=Etc/UTC

WORKDIR /app

# Copy virtual environment (with the installed package) from the builder
COPY --chown=1000:1000 --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Switch to the pre-existing UID 1000 non root user
RUN chown 1000:1000 /app
USER 1000

ENTRYPOINT ["ska-sdp-instrumental-calibration"]
