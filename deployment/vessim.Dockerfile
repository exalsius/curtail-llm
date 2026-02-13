# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/exalsius/pilot"
LABEL org.opencontainers.image.description="Vessim energy simulation for DERMS pilot"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "vessim[sil]>=0.13.1" "pandas>=2.1.0"
    
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --chown=app:app mci.csv .
COPY --chown=app:app energy_simulation.py .

RUN mkdir -p results && chown -R app:app results

USER app

EXPOSE 8800

ENTRYPOINT ["python", "-u", "energy_simulation.py"]
