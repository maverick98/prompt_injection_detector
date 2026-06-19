# Streamlit Cloud Deployment

Use these settings in Streamlit Community Cloud.

| Field | Value |
|---|---|
| Repository | `maverick98/prompt_injection_detector` |
| Branch | `master` |
| Main file path | `streamlit_app.py` |
| Python version | default / latest supported |

The repo includes:

- `requirements.txt` for Streamlit Cloud dependency installation
- `streamlit_app.py` as the root app entrypoint
- `.streamlit/config.toml` for app theme and server config

The app trains a lightweight local detector on first startup if no serialized
artifact is available in the deployment environment. This keeps deployment simple
because generated model artifacts do not need to be committed to Git.

