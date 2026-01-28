FROM flwr/superexec:1.25.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app
COPY --chown=app:app pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
  && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flower-superexec"]
