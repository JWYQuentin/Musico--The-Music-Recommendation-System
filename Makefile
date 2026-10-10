.PHONY: setup download convert subsample audit phase1 split baselines-tune baselines features retrieval coldstart refit candidates ranker-tune ranker test

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

features:
	python -m m4a_rec.features

retrieval:
	python -m m4a_rec.retrieval report

coldstart:
	python -m m4a_rec.coldstart report

refit:
	python -m m4a_rec.baselines refit
	python -m m4a_rec.retrieval refit

candidates:
	python -m m4a_rec.candidates

ranker-tune:
	python -m m4a_rec.ranker tune
	python -m m4a_rec.ranker ablate

ranker:
	python -m m4a_rec.ranker report

test:
	pytest -q
