# Public deployment

Dashboard: https://f1-race-forecast.streamlit.app/

Repository: https://github.com/Amr-Belal-77/f1-race-forecast

## Streamlit Community Cloud configuration

- Repository: `Amr-Belal-77/f1-race-forecast`
- Branch: `main`
- Entry point: `app.py`
- Python: `3.11`
- Custom subdomain: `f1-race-forecast`
- Access: public, with no viewer sign-in required
- Secrets: none required

The dashboard starts from committed `reports/demo/` files, so deployment does not download race history, train models, or load joblib artifacts. The root `requirements.txt` installs the local package and runtime dependencies.

Changes pushed to `main` are picked up by Community Cloud. Update all four demo report files together after a full pipeline run to preserve their checksums. Do not upload virtual environments, API caches, local model binaries or secrets.

Use the app's management panel in Community Cloud to view build logs, reboot the app or change its settings. The public app runs on the hosting service and does not depend on the local development computer remaining online.

Reference: [Streamlit Community Cloud deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
