# Local demo walkthrough

1. Install Python 3.11, create a virtual environment, and install `requirements-demo.txt`.
2. From the repository root run `python serve.py`.
3. Open http://127.0.0.1:8080. Confirm the API-online indicator appears.
4. Click **Load synthetic demo image**. Inspect the synthetic label and preview.
5. Click **Run classification**. The first run loads the bundled model. The result shows its real output and raw, uncalibrated softmax scores.
6. Click **Remove** and upload a JPEG or PNG. Files are handled in memory. The model cannot determine whether the image is medically appropriate.
7. To check readiness directly, visit http://127.0.0.1:8080/ready. A 503 indicates the model could not load.

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
