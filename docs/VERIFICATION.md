# Verification record â€” 2026-10-29

## Story verified

A user opens the local product, loads a synthetic image or selects a file, submits it to Flask, runs the bundled VGG16 model, and sees its actual class and raw softmax scores. Invalid uploads surface a readable error and the user can recover with another file.

## Environment and artifact

- Windows host; Python **3.11.9**; TensorFlow **2.15.0**; Flask **3.1.2**; Waitress **3.0.2**; Pillow **11.3.0**; NumPy **1.26.4**.
- Served with `python serve.py` at `http://127.0.0.1:8080`.
- Local virtual environment reused system packages; TensorFlow/Keras were reinstalled into that environment after discovering the existing TensorFlow installation was incomplete. The committed direct dependencies define a standalone serving setup; CI tests it separately.
- Model SHA-256: `cd69c3fe47bc0eea916d99ac2a73c5baaf3f8021c95ead8318ec77b628498a6c`.
- Model weights were not changed or retrained.
- Input: the generated `static/synthetic-demo.png` fixture. No patient images were used or published.

## Checks and evidence

| Boundary | Result | Evidence |
| --- | --- | --- |
| UI renders | Passed | Desktop screenshot; all assets served locally; no external font/script dependencies |
| Image selection | Passed | Built-in sample and native file chooser both display a preview |
| Browser â†’ API â†’ model | Passed | Browser Run classification returned Normal, raw scores Normal 0.8590835333 / Tumor 0.1409164369 |
| Response â†’ UI | Passed | Actual scores displayed as 85.91% / 14.09%; repeated browser run completed in 0.21 s |
| Invalid input â†’ error | Passed | Corrupt PNG returned a readable API error shown in the browser |
| Error recovery | Passed | Remove, upload a valid file, and run inference successfully |
| Responsive view | Passed | 390 Ã— 844 viewport; document scroll width 375, no horizontal overflow; completed result visible in full-page screenshot |
| Health / readiness | Passed | Actual HTTP 200 responses from both endpoints |
| Web training blocked | Passed | Actual HTTP 403 from `/train` |
| Concurrent requests | Passed | Three real HTTP predictions completed with identical fixture scores |
| Automated suite | Passed | `python -m pytest -q`: **20 passed, 1 warning in 11.72 s** |

[Saved HTTP evidence](live-test-results.json) includes the actual responses, timings, UTC check timestamp, and model hash. HTTP fixture inference took **0.144 s** after model loading; three simultaneous requests completed in **0.151â€“0.414 s**. These timings describe this local run only and are not load-test or service-level guarantees. The first browser inference took **42.20 s**, including TensorFlow/model initialization during environment setup.

The automated suite covers missing/malformed JSON, wrong content type, invalid base64, unsupported/corrupt images, request/image/pixel limits, normalization, RGB/grayscale/RGBA conversion, JPEG/PNG, security headers, unavailable models, training rejection, real-model responses, repeated inference, and concurrent requests with different input images. Validation-only cases use a controlled fake engine; the actual-model test uses the bundled HDF5 artifact.

The local warning comes from an inherited `requests` dependency-version combination in the shared system environment. TensorFlow also prints legacy API deprecation notices during startup. Neither prevented the recorded successful execution; a clean deployment should use the standalone serving dependencies rather than a shared system environment.

## Screenshots

1. [Desktop workspace](showcase/01-desktop.jpg)
2. [Image preview](showcase/02-image-preview.jpg)
3. [Real inference result](showcase/03-live-result.jpg)
4. [Mobile result](showcase/04-mobile.jpg)
5. [Validation error](showcase/05-validation.jpg)

The synthetic fixture is intentionally labeled. Its output is a software demonstration, not medical evidence. The screenshots were captured from the running local application rather than rendered mockups.

## Scope and remaining limits

- **Medical performance:** not evaluated. The historical 48.20% accuracy remains unchanged and cannot be attributed conclusively to this serving artifact. No patient dataset, clinical evaluation, new training, calibration, or diagnostic validation was performed.
- **Container:** Docker CLI was present, but the Docker engine was unavailable. The updated Dockerfile was not built or run locally.
- **Public hosting:** no internet-facing inference service was deployed. The live demo is local and uses a production WSGI server. TLS, access control, rate limiting, observability, deployment testing, and load testing remain deployment responsibilities.
- **Training:** the original four-stage pipeline was inspected but not rerun; dataset access/rights and training environment reproducibility remain unverified.
- **Browser console:** no warnings/errors were returned by the browser log inspection during the tested flow. Invalid-image error handling was confirmed through the visible UI and HTTP status.
- **CI:** the new `Demo checks` workflow runs the same suite on Ubuntu/Python 3.11. See the repository Actions tab for the actual run outcome; local checks alone do not imply CI passed.
