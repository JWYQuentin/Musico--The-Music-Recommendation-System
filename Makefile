.PHONY: setup download convert subsample audit phase1 split baselines-tune baselines test

setup:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

download:
	python -m m4a_rec.download

convert:
	python -m m4a_rec.prepare convert

subsample:
	python -m m4a_rec.prepare subsample

audit:
	python -m m4a_rec.audit

phase1: download convert subsample audit

split:
	python -m m4a_rec.split

baselines-tune:
	python -m m4a_rec.baselines tune

baselines:
	python -m m4a_rec.baselines report

test:
	pytest -q
