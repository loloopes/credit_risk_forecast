#!/usr/bin/env bash
set -euo pipefail

# Sync local Airflow credentials file to GitHub Actions secrets.
# Requires: gh CLI authenticated with repo admin/maintainer permissions.

ENV_FILE="${1:-.github/secrets/airflow-secrets.env}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) is not installed."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Error: gh is not authenticated. Run: gh auth login"
  exit 1
fi

if [ ! -f "${ENV_FILE}" ]; then
  echo "Error: '${ENV_FILE}' not found."
  echo "Create it from '.github/secrets/airflow-secrets.env.example'."
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required_vars=(
  AIRFLOW_API_URL
  AIRFLOW_API_USERNAME
  AIRFLOW_API_PASSWORD
)

for var_name in "${required_vars[@]}"; do
  if [ -z "${!var_name:-}" ]; then
    echo "Error: ${var_name} is empty in '${ENV_FILE}'."
    exit 1
  fi
done

printf "%s" "${AIRFLOW_API_URL}" | gh secret set AIRFLOW_API_URL
printf "%s" "${AIRFLOW_API_USERNAME}" | gh secret set AIRFLOW_API_USERNAME
printf "%s" "${AIRFLOW_API_PASSWORD}" | gh secret set AIRFLOW_API_PASSWORD

echo "GitHub Actions secrets updated successfully."
