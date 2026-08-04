.PHONY: data gold train evaluate test lint typecheck build serve all

data: data/raw/_manifest.json data/bronze/_manifest.json data/silver/_manifest.json

data/raw/_manifest.json:
	mtpl ingest

data/bronze/_manifest.json: data/raw/_manifest.json
	mtpl bronze

data/silver/_manifest.json: data/bronze/_manifest.json
	mtpl silver

gold: data/gold/frequency/_manifest.json data/gold/severity/_manifest.json data/gold/pure_premium/_manifest.json

data/gold/frequency/_manifest.json: data/silver/_manifest.json
	mtpl gold --use-case frequency

data/gold/severity/_manifest.json: data/silver/_manifest.json
	mtpl gold --use-case severity

data/gold/pure_premium/_manifest.json: data/silver/_manifest.json
	mtpl gold --use-case pure_premium

train: gold
	mtpl train frequency --register-as mtpl_frequency
	mtpl train severity --register-as mtpl_severity
	mtpl train tweedie --register-as mtpl_pure_premium
	mtpl train challenger --use-case frequency
	mtpl train challenger --use-case pure_premium

evaluate:
	mtpl evaluate --run-id $(RUN_ID)

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy src/

build:
	docker build -t mtpl-pricing-engine .

serve:
	mtpl serve

all: data gold train test
