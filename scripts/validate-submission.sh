#!/usr/bin/env bash

set -euo pipefail

DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-600}"
DOCKER_RUN_TIMEOUT="${DOCKER_RUN_TIMEOUT:-45}"
DEFAULT_LOCAL_PORT="${LOCAL_PORT:-18000}"

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BOLD=''
  NC=''
fi

log_info() {
  printf "%b[INFO]%b %s\n" "${YELLOW}" "${NC}" "$*"
}

log_ok() {
  printf "%b[OK]%b %s\n" "${GREEN}" "${NC}" "$*"
}

log_fail() {
  printf "%b[FAIL]%b %s\n" "${RED}" "${NC}" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/validate-submission.sh <space_base_url> [repo_dir]

Examples:
  ./scripts/validate-submission.sh https://reteesh-rl-finance-manager.hf.space
  ./scripts/validate-submission.sh https://reteesh-rl-finance-manager.hf.space /path/to/repo

What it checks:
  1. Required files exist
  2. Hugging Face Space /health responds
  3. openenv validate passes
  4. Docker image builds
  5. Docker container starts and local /health responds
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log_fail "Missing required command: $1"
    exit 1
  fi
}

cleanup() {
  if [ -n "${CONTAINER_ID:-}" ]; then
    docker rm -f "${CONTAINER_ID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

SPACE_BASE_URL="${1:-}"
REPO_DIR="${2:-$(pwd)}"

if [ -z "${SPACE_BASE_URL}" ]; then
  usage
  exit 1
fi

SPACE_BASE_URL="${SPACE_BASE_URL%/}"

if [ ! -d "${REPO_DIR}" ]; then
  log_fail "Repository directory does not exist: ${REPO_DIR}"
  exit 1
fi

if [ -f "${REPO_DIR}/openenv.yaml" ]; then
  ENV_DIR="${REPO_DIR}"
elif [ -f "${REPO_DIR}/rl_finance/openenv.yaml" ]; then
  ENV_DIR="${REPO_DIR}/rl_finance"
else
  log_fail "Could not find openenv.yaml in ${REPO_DIR} or ${REPO_DIR}/rl_finance"
  exit 1
fi

require_cmd curl
require_cmd docker
require_cmd timeout

if [ -x "${ENV_DIR}/.venv/bin/openenv" ]; then
  OPENENV_CMD="${ENV_DIR}/.venv/bin/openenv"
elif command -v openenv >/dev/null 2>&1; then
  OPENENV_CMD="$(command -v openenv)"
else
  log_fail "Could not find openenv. Install it or create ${ENV_DIR}/.venv first."
  exit 1
fi

log_info "Using environment directory: ${ENV_DIR}"

for path in \
  "${ENV_DIR}/openenv.yaml" \
  "${ENV_DIR}/Dockerfile" \
  "${ENV_DIR}/pyproject.toml" \
  "${REPO_DIR}/README.md" \
  "${REPO_DIR}/inference.py"; do
  if [ ! -e "${path}" ]; then
    log_fail "Missing required file: ${path}"
    exit 1
  fi
done
log_ok "Required files found"

log_info "Checking Hugging Face Space health: ${SPACE_BASE_URL}/health"
if curl -fsS --max-time 15 "${SPACE_BASE_URL}/health" >/tmp/rl_finance_space_health.json; then
  log_ok "Space health endpoint responded"
else
  log_fail "Space health endpoint did not respond successfully"
  exit 1
fi

log_info "Running openenv validate"
(
  cd "${ENV_DIR}"
  "${OPENENV_CMD}" validate
)
log_ok "openenv validate passed"

IMAGE_TAG="rl-finance-precheck:latest"
log_info "Building Docker image ${IMAGE_TAG}"
(
  cd "${ENV_DIR}"
  timeout "${DOCKER_BUILD_TIMEOUT}" docker build -t "${IMAGE_TAG}" .
)
log_ok "Docker build passed"

log_info "Starting container and checking local /health"
CONTAINER_ID="$(
  docker run -d -p "${DEFAULT_LOCAL_PORT}:8000" "${IMAGE_TAG}"
)"

healthy=0
for _ in $(seq 1 "${DOCKER_RUN_TIMEOUT}"); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${DEFAULT_LOCAL_PORT}/health" >/tmp/rl_finance_local_health.json; then
    healthy=1
    break
  fi
  sleep 1
done

if [ "${healthy}" -ne 1 ]; then
  log_fail "Local container health check failed"
  docker logs "${CONTAINER_ID}" || true
  exit 1
fi

log_ok "Local container health endpoint responded"

printf "\n%bValidation completed successfully.%b\n" "${BOLD}${GREEN}" "${NC}"
printf "Space URL: %s\n" "${SPACE_BASE_URL}"
printf "Environment directory: %s\n" "${ENV_DIR}"
