FROM artefact.skao.int/ska-build-python:0.3.3 AS build

WORKDIR /build

COPY . ./

ENV POETRY_VIRTUALENVS_CREATE=false

RUN poetry install --no-root --only main \
    && pip install --no-compile --no-cache-dir --no-dependencies .

FROM artefact.skao.int/ska-python:0.2.5

WORKDIR /app

COPY --from=build /usr/local/ /usr/local/

ENTRYPOINT ["ska-sdp-instrumental-calibration"]