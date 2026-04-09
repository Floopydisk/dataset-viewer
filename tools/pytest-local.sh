#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_FILE="$ROOT_DIR/tools/docker-compose-mongo.yml"

print_usage() {
  cat <<'USAGE'
Usage:
  tools/pytest-local.sh [--keep-mongo] [--no-install] <component> [pytest args...]

Components:
  libcommon   Run tests in libs/libcommon with local Mongo on port 27020
  api         Run tests in services/api with local Mongo on port 27031

Options:
  --keep-mongo  Keep Mongo test dependencies running after pytest exits
  --no-install  Skip poetry install even if .venv is missing

Examples:
  tools/pytest-local.sh libcommon tests/test_train.py -q
  tools/pytest-local.sh api tests/routes/test_train.py -q
  tools/pytest-local.sh --keep-mongo libcommon tests/test_train.py -q
USAGE
}

KEEP_MONGO="false"
NO_INSTALL="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-mongo)
      KEEP_MONGO="true"
      shift
      ;;
    --no-install)
      NO_INSTALL="true"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "error: unknown option '$1'" >&2
      print_usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

component="$1"
shift

if command -v poetry@2.1.4 >/dev/null 2>&1; then
  POETRY_BIN="poetry@2.1.4"
elif command -v poetry >/dev/null 2>&1; then
  POETRY_BIN="poetry"
else
  echo "error: Poetry is not installed. Install Poetry first." >&2
  exit 1
fi

case "$component" in
  libcommon)
    PROJECT_DIR="$ROOT_DIR/libs/libcommon"
    CACHE_MONGO_URL="mongodb://localhost:27020"
    QUEUE_MONGO_URL="mongodb://localhost:27020"
    ;;
  api)
    PROJECT_DIR="$ROOT_DIR/services/api"
    CACHE_MONGO_URL="mongodb://localhost:27031"
    QUEUE_MONGO_URL="mongodb://localhost:27031"
    ;;
  *)
    echo "error: unsupported component '$component'" >&2
    print_usage
    exit 1
    ;;
esac

echo "[pytest-local] Starting Mongo test dependencies..."
if [[ "$KEEP_MONGO" == "true" ]]; then
  docker compose -f "$DOCKER_COMPOSE_FILE" up -d --wait --wait-timeout 20 >/dev/null
else
  docker compose -f "$DOCKER_COMPOSE_FILE" up -d --build --force-recreate --remove-orphans --renew-anon-volumes --wait --wait-timeout 20 >/dev/null
fi

cleanup() {
  echo "[pytest-local] Stopping Mongo test dependencies..."
  docker compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans --volumes >/dev/null
}

if [[ "$KEEP_MONGO" != "true" ]]; then
  trap cleanup EXIT
fi

cd "$PROJECT_DIR"

if [[ "$NO_INSTALL" != "true" && ! -d .venv ]]; then
  echo "[pytest-local] Installing dependencies for $component..."
  "$POETRY_BIN" install
fi

echo "[pytest-local] Running pytest in $PROJECT_DIR"
CACHE_MONGO_URL="$CACHE_MONGO_URL" \
QUEUE_MONGO_URL="$QUEUE_MONGO_URL" \
"$POETRY_BIN" run python -m pytest -vv -x "$@"

if [[ "$KEEP_MONGO" == "true" ]]; then
  echo "[pytest-local] Mongo test dependencies are still running."
  echo "[pytest-local] Stop them with: docker compose -f tools/docker-compose-mongo.yml down --remove-orphans --volumes"
fi
