## Airflow Secrets for GitHub Actions

These values are required by `.github/workflows/ci-cd.yml` to trigger the retraining DAG from GitHub Actions:

- `AIRFLOW_API_URL` — can be a **Repository variable** (`Settings → Secrets and variables → Actions → Variables`) or a **secret** with the same name.
- `AIRFLOW_API_USERNAME` — **secret** (recommended).
- `AIRFLOW_API_PASSWORD` — **secret** (recommended).

If all three are empty in the workflow log, they were not defined for **this** repository (or the run is from a fork without secrets).

### Quick setup

1. Copy the template:

```bash
cp .github/secrets/airflow-secrets.env.example .github/secrets/airflow-secrets.env
```

2. Edit `.github/secrets/airflow-secrets.env` with your real values.

3. Sync values to GitHub repository secrets:

```bash
bash scripts/sync-github-airflow-secrets.sh
```

### Optional custom env path

```bash
bash scripts/sync-github-airflow-secrets.sh /path/to/airflow-secrets.env
```

### Notes

- Keep `.github/secrets/airflow-secrets.env` private; do not commit it.
- The self-hosted runner still needs network access to `AIRFLOW_API_URL` (VPN/tunnel).
- **Airflow 3** public REST API uses **JWT** (`POST /auth/token` then `Authorization: Bearer ...`). HTTP Basic auth against `/api/v2/...` returns **401**.
