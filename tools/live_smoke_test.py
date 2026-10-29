"""Exercise the running service with synthetic fixtures and save HTTP evidence."""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8080'
fixture = (ROOT / 'static/synthetic-demo.png').read_bytes()
payload = {'image': base64.b64encode(fixture).decode()}


def call(path, body=None, expected=200):
    started = time.perf_counter()
    request = Request(BASE + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={} if body is None else {'Content-Type': 'application/json'})
    try:
        response = urlopen(request, timeout=180)
    except HTTPError as exc:
        response = exc
    with response:
        status = response.status
        data = json.load(response)
    assert status == expected, (path, status, data)
    return {'route': path, 'status': status, 'seconds': round(time.perf_counter()-started, 3), 'response': data}


results = [call('/health'), call('/ready'), call('/predict', payload),
           call('/predict', {'image': 'invalid!!!'}, 400),
           call('/predict', {'image': base64.b64encode(b'corrupt').decode()}, 400),
           call('/train', expected=403)]
with ThreadPoolExecutor(max_workers=3) as executor:
    concurrent = list(executor.map(lambda _: call('/predict', payload), range(3)))
assert all(item['response'][0]['scores'] == results[2]['response'][0]['scores'] for item in concurrent)
report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
          'base_url': BASE, 'input': 'synthetic-demo.png; no patient data',
          'model_sha256': hashlib.sha256((ROOT / 'model/model.h5').read_bytes()).hexdigest(),
          'checks': results, 'concurrent_predictions': concurrent,
          'scope': 'Software execution only; not medical accuracy or load validation.'}
(ROOT / 'docs/live-test-results.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
