"""Dependency-free Rspamd health check; optional synthetic protocol/GTUBE smoke test."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

GTUBE = 'XJS*C4JDBQADN1.NSBN3*2IDNEN*GTUBE-STANDARD-ANTI-UBE-TEST-EMAIL*C.34X'


def validate_scan(result):
    if not isinstance(result, dict) or result.get('error') is not None:
        raise ValueError('Rspamd scan returned an error or invalid object')
    if not isinstance(result.get('action'), str) or not result['action'].strip():
        raise ValueError('Rspamd action is missing')
    for key in ('score', 'required_score'):
        if isinstance(result.get(key), bool) or not isinstance(result.get(key), (int, float, str)):
            raise ValueError('Rspamd numeric field is missing or invalid: ' + key)
        if not math.isfinite(float(result[key])):
            raise ValueError('Rspamd numeric field is not finite: ' + key)
    symbols = result.get('symbols', {})
    if not isinstance(symbols, dict):
        raise ValueError('Rspamd symbols must be an object')
    return dict(action=result['action'], score=float(result['score']),
                required_score=float(result['required_score']), symbol_count=len(symbols),
                is_skipped=result.get('is_skipped'), passthrough_module=result.get('passthrough_module'),
                gtube_detected='GTUBE' in symbols)


def check(url, smoke_test=False, timeout=10):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use an HTTP(S) Rspamd URL without embedded credentials, query or fragment')
    headers = {}
    if os.environ.get('RSPAMD_PASSWORD'):
        headers['Password'] = os.environ['RSPAMD_PASSWORD']
    # Direct access is restricted to loopback; other destinations retain inherited proxy policy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if parsed.hostname in ('localhost', '127.0.0.1', '::1') else urllib.request.build_opener()
    def request(endpoint, body=None):
        request_headers = dict(headers)
        if body is not None:
            request_headers['Content-Type'] = 'message/rfc822'
        req = urllib.request.Request(url.rstrip('/') + '/' + endpoint, data=body, headers=request_headers)
        with opener.open(req, timeout=timeout) as response:
            data = response.read(4_000_001)
        if len(data) > 4_000_000:
            raise ValueError('Rspamd response exceeded the supported size')
        return data
    started = time.monotonic()
    if request('ping').strip().lower() != b'pong':
        raise ValueError('Rspamd ping did not return pong')
    result = dict(status='ok', ping_ok=True, smoke_test_performed=smoke_test,
                  quality_test_performed=False, client_mail_used=False, learn_endpoint_called=False)
    if smoke_test:
        # Isolate the synthetic protocol test from greylisting prefilter state.
        # This is request-scoped and does not alter the server configuration.
        headers['Settings'] = json.dumps({'symbols_disabled': ['GREYLIST_CHECK', 'GREYLIST_SAVE']})
        result['greylisting_disabled_for_synthetic_requests'] = True
        prefix = b'From: test@example.invalid\r\nTo: test@example.invalid\r\nSubject: MailGuard synthetic smoke test\r\n\r\n'
        result['synthetic_plain'] = validate_scan(json.loads(request('checkv2', prefix + b'Engineering check only.\r\n')))
        result['synthetic_gtube'] = validate_scan(json.loads(request('checkv2', prefix + GTUBE.encode('ascii') + b'\r\n')))
        if not result['synthetic_gtube']['gtube_detected']:
            raise ValueError('GTUBE was not detected; inspect whether the test symbol is disabled')
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('RSPAMD_URL', 'http://127.0.0.1:11333/'))
    parser.add_argument('--smoke-test', action='store_true', help='Submit two synthetic messages with request-scoped greylisting disabled; may affect server statistics/autolearn')
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Timeout must be positive and finite')
    try:
        result = check(args.url, args.smoke_test, args.timeout)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        # Do not echo remote response bodies, credential headers or client messages.
        result = dict(status='failed', error_type=type(exc).__name__, quality_test_performed=False)
    text = json.dumps(result, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text + '\n')
    print(text)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    raise SystemExit(main())
