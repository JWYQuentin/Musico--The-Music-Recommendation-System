.PHONY: setup download convert subsample audit phase1 split baselines-tune baselines features retrieval coldstart refit candidates ranker-tune ranker metadata bias-tune bias demo screenshots test

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

metadata:
	python -m m4a_rec.metadata

ranker-tune:
	python -m m4a_rec.ranker tune ranker_artist
	python -m m4a_rec.ranker ablate ranker_artist

ranker:
	python -m m4a_rec.ranker fit ranker
	python -m m4a_rec.ranker fit ranker_artist
	python -m m4a_rec.ranker report

bias-tune:
	python -m m4a_rec.bias tune

bias:
	python -m m4a_rec.bias report

demo:
	python -m m4a_rec.index data/processed/models/tt_id_refit
	python -m m4a_rec.demo

# Pictures of the demo for the README, taken with headless Chrome in the light theme.
# They show a few track titles from Music4All: look at them before committing.
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
DEMO = file://$(CURDIR)/data/processed/demo/index.html
screenshots:
	"$(CHROME)" --headless=new --hide-scrollbars --window-size=1440,1250 --screenshot=reports/figures/demo_listener.png "$(DEMO)#theme=light"
	"$(CHROME)" --headless=new --hide-scrollbars --window-size=1440,820 --screenshot=reports/figures/demo_track.png "$(DEMO)#theme=light&t=first"

test:
	pytest -q
