FROM python:3.11-slim AS builder

WORKDIR /build
RUN python -m venv /venv
ENV PATH="/venv/bin:${PATH}"

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
RUN pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

ARG GIT_SHA=unknown
ENV MTPL_GIT_SHA=${GIT_SHA}

RUN useradd --create-home --uid 1000 mtpl
WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:${PATH}"

COPY src ./src
COPY api ./api
COPY artifacts ./artifacts
COPY mlruns ./mlruns

# mlflow's local file store bakes the absolute path it was written under into
# every meta.yaml; without this the registry points back at the build host.
RUN python -c "\
import pathlib, re; \
[p.write_text(re.sub(r'file:///[^\"\n]*?/mlruns', 'file:///app/mlruns', p.read_text())) \
 for p in pathlib.Path('mlruns').rglob('meta.yaml')]"

RUN chown -R mtpl:mtpl /app
USER mtpl

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
