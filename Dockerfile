FROM artefact.skao.int/ska-base-ubuntu24:1.1.2 AS build

WORKDIR /build

COPY . ./

ENV POETRY_VIRTUALENVS_CREATE=false

RUN poetry install --no-root --only main \
    && pip install --no-compile --no-cache-dir --no-dependencies .

FROM artefact.skao.int/ska-python-ubuntu24:1.1.2

WORKDIR /app

COPY --from=build /usr/local/ /usr/local/

ENTRYPOINT ["ska-sdp-instrumental-calibration"]