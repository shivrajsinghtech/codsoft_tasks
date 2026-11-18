# Security Policy

## Reporting a vulnerability

Please report security issues **privately** rather than in a public issue: use the repository's
**Security → Report a vulnerability** button (GitHub private vulnerability reporting). Include steps
to reproduce and the affected file or version.

This is a student internship project, so fixes are made on a best-effort basis.

## Safe use of the model file

`models/genre_classifier.joblib` is a pickle-based file. **Loading a pickle can execute code**, so:

- only load model files you created yourself or downloaded from this repository;
- never pass an untrusted file to `predict.py --model-path` or `MOVIE_GENRE_MODEL_PATH`.

## Secrets

The project needs no API keys or credentials. Never commit `.env` files or Streamlit
`secrets.toml` (both are listed in `.gitignore`).

## Dependency checks

Versions are pinned in `requirements.txt`. To check them for known vulnerabilities:

```bash
pip install pip-audit && pip-audit -r requirements.txt
```
