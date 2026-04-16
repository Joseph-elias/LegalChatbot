#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/opt/legalchatbot}
COMPOSE_FILES=${COMPOSE_FILES:-docker-compose.yml,docker-compose.internet.yml}
COMPOSE_ENV_FILE=${COMPOSE_ENV_FILE:-deploy/.env.internet}
cd "$REPO_DIR"

git fetch --all
git reset --hard "origin/${BRANCH:-main}"

compose_args=()
IFS=',' read -r -a compose_files <<< "$COMPOSE_FILES"
for f in "${compose_files[@]}"; do
  compose_args+=(-f "$f")
done
if [ -n "$COMPOSE_ENV_FILE" ] && [ -f "$COMPOSE_ENV_FILE" ]; then
  compose_args=(--env-file "$COMPOSE_ENV_FILE" "${compose_args[@]}")
fi

docker compose "${compose_args[@]}" pull || true
docker compose "${compose_args[@]}" build --pull
docker compose "${compose_args[@]}" up -d --remove-orphans

docker image prune -f
