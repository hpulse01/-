#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

# Backend licenses
if command -v pip-licenses >/dev/null 2>&1; then
  (cd backend && pip-licenses --format=json --with-authors --with-urls --with-license-file > ../THIRD_PARTY_BACKEND.json)
else
  echo "pip-licenses not found. Installing..." >&2
  python -m pip install pip-licenses
  (cd backend && pip-licenses --format=json --with-authors --with-urls --with-license-file > ../THIRD_PARTY_BACKEND.json)
fi

# Frontend licenses
if [ -d frontend/node_modules ]; then
  npx license-checker --production --json > THIRD_PARTY_FRONTEND.json
else
  echo "node_modules not found in frontend. Run npm ci first." >&2
fi

echo "Generated THIRD_PARTY_BACKEND.json and THIRD_PARTY_FRONTEND.json at project root."