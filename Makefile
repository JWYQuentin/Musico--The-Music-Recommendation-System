.PHONY: setup download convert subsample audit phase1 split baselines-tune baselines features retrieval coldstart refit candidates ranker-tune ranker metadata bias-tune bias demo screenshots newcomers bundle serving serve docker-build docker-run test

setup:
	python -m venv .venv && .venv/bin/pip install -e ".[train,serve,dev]"

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

# Pictures of the demo, taken with headless Chrome in the light theme. They show titles
# and artists from Music4All, so they stay beside the page in data/, which git ignores.
# The README's pictures are of an invented listener instead.
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
DEMO = file://$(CURDIR)/data/processed/demo/index.html
screenshots:
	"$(CHROME)" --headless=new --hide-scrollbars --window-size=1440,1250 --screenshot=data/processed/demo/demo_listener.png "$(DEMO)#theme=light"
	"$(CHROME)" --headless=new --hide-scrollbars --window-size=1440,820 --screenshot=data/processed/demo/demo_track.png "$(DEMO)#theme=light&t=first"

# Phase 7: the service. `newcomers` trains for about 30 minutes.
newcomers:
	python -m m4a_rec.newcomers
	python -m m4a_rec.bundle newcomers

bundle:
	python -m m4a_rec.bundle

serving:
	python -m m4a_rec.serve report

serve:
	python -m m4a_rec.api

# The image holds code only. The bundle is mounted read-only, and the port is published on
# this machine only: the responses carry Music4All titles.
docker-build:
	docker build -t m4a-rec .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 -v $(CURDIR)/data/processed/serving:/app/data/processed/serving:ro m4a-rec

test:
	pytest -q
