PY := venv/bin/python

.PHONY: all raw bronze silver gold clean rebuild

all: gold

raw: data/raw/_manifest.json
data/raw/_manifest.json: data/raw/raw.py
	$(PY) data/raw/raw.py

bronze: data/bronze/_manifest.json
data/bronze/_manifest.json: data/bronze/bronze.py data/raw/_manifest.json
	$(PY) data/bronze/bronze.py

silver: data/silver/_manifest.json
data/silver/_manifest.json: data/silver/silver.py data/bronze/_manifest.json
	$(PY) data/silver/silver.py

gold: data/gold/_manifest.json
data/gold/_manifest.json: data/gold/gold.py data/silver/_manifest.json
	$(PY) data/gold/gold.py

# Delete everything downstream of raw. Raw itself is left alone: it's an
# external extract pinned by OpenML dataset id + version, not a derived
# artifact, so there's nothing to "rebuild" about it beyond re-fetching.
clean:
	rm -f data/bronze/*.parquet data/bronze/_manifest.json \
	      data/silver/*.parquet data/silver/_manifest.json \
	      data/gold/*.parquet data/gold/_manifest.json

rebuild: clean all
