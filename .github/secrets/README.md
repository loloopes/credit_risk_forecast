## Airflow Secrets for GitHub Actions

These secrets are required by `.github/workflows/ci-cd.yml` to trigger the retraining DAG from GitHub Actions:

- `AIRFLOW_API_URL`
- `AIRFLOW_API_USERNAME`
- `AIRFLOW_API_PASSWORD`

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
