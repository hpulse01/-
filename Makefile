SHELL := /usr/bin/bash

COMPOSE := docker compose
ENV_FILE := .env

.PHONY: up down logs build rebuild restart backend frontend ps test fmt lint migrate seed alembic heads

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f --tail=200

build:
	$(COMPOSE) build

rebuild:
	$(COMPOSE) build --no-cache

restart:
	$(COMPOSE) restart

backend:
	$(COMPOSE) up -d backend

frontend:
	$(COMPOSE) up -d frontend

ps:
	$(COMPOSE) ps

test:
	$(COMPOSE) run --rm backend pytest -q

fmt:
	$(COMPOSE) run --rm backend bash -lc "ruff format . && ruff check --fix ."
	$(COMPOSE) run --rm frontend npm run format

lint:
	$(COMPOSE) run --rm backend bash -lc "ruff check . && mypy backend/app || true"
	$(COMPOSE) run --rm frontend npm run lint

migrate:
	$(COMPOSE) run --rm backend alembic upgrade head

seed:
	$(COMPOSE) run --rm backend bash scripts/seed.sh