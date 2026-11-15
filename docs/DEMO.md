# Local demo walkthrough

1. Install Python 3.11, create a virtual environment, and install `requirements-demo.txt`.
2. From the repository root run `python serve.py`.
3. Open http://127.0.0.1:8080. Confirm the API-online indicator appears.
4. Check the active model under the page title. The app selects the four-class CNN if its checkpoint is present at startup and VGG16 otherwise.
5. Click **Try a synthetic sample**. Inspect its label and preview.
6. Click **Run analysis**. The first run loads the active model. Review the predicted category, image details, and raw scores. Use **Show original** to compare the image with its attention overlay when available.
7. Click **Remove** and upload a JPEG or PNG. Files are handled in memory. The model cannot determine whether the image is medically appropriate.
8. To check readiness directly, visit http://127.0.0.1:8080/ready. A 503 indicates the model could not load.

The generated fixture is a deterministic CT-like geometric pattern, not a scan. No label from that fixture has a clinical interpretation. Regenerate it with `python tools/generate_demo_fixture.py`.

## Troubleshooting

- **Wrong Python:** use Python 3.11 for the pinned serving dependencies; Python 3.14 does not match this environment.
- **Model unavailable:** confirm `model/model.h5` exists and inspect server logs. Do not load untrusted replacement model files.
- **Initial delay:** model loading can take longer than subsequent inference. Browser requests time out after 120 seconds; model readiness can be checked before a demo.
- **Port in use:** set `PORT` to another free port before starting the server.
- **Training environment conflicts:** use separate virtual environments for `requirements-demo.txt` and the original `requirements.txt`.
- **Public hosting:** this walkthrough runs locally. Provisioning an internet-facing service requires deployment infrastructure and operational controls beyond this demo.

## Reproducible checks

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Run the server separately and use `python tools/live_smoke_test.py` to collect actual HTTP request/response evidence in `docs/live-test-results.json`. That script sends only synthetic fixtures to loopback by default.
