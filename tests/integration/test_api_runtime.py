"""Linux/local API contract tests; synthetic .mg fixture, never production weights."""
import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOTNET = os.environ.get('MAILGUARD_TEST_DOTNET') or shutil.which('dotnet')
API = ROOT / 'src/MailGuard.Api/bin/Release/net8.0/MailGuard.Api.dll'


class Backend(BaseHTTPRequestHandler):
    mode = 'valid'
    last_type = None

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length', 0)))
        Backend.last_type = self.headers.get('Content-Type')
        if self.mode == 'slow':
            time.sleep(2)
        payload = {'action': 'no action', 'score': 0, 'required_score': 6}
        if self.mode == 'deferred':
            payload['action'] = 'soft reject'
        if self.mode == 'invalid':
            payload = {'error': 'private-canary-do-not-log'}
        self.send_response(502 if self.mode == 'http_error' else 200)
        self.send_header('Content-Type', 'application/json'); self.end_headers()
        try:
            self.wfile.write(json.dumps(payload).encode())
        except BrokenPipeError:
            pass

    def log_message(self, *args):
        pass


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextlib.contextmanager
def api_process(backend_port, model=True):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'fixture.mg'
        if model:
            def string(value):
                raw = value.encode(); return bytes([len(raw)]) + raw
            path.write_bytes(string('MAILGUARD043') + string('integration-fixture') +
                struct.pack('<iddd?', 2048, -20., .8, .2, True) + bytes(2048 * 4) + struct.pack('<ii', 0, 0))
        api_port = port()
        env = dict(os.environ, MAILGUARD_MODEL=str(path), RSPAMD_URL=f'http://127.0.0.1:{backend_port}/',
                   MAILGUARD_MAX_MESSAGE_BYTES='256', RSPAMD_TIMEOUT_SECONDS='1',
                   ASPNETCORE_URLS=f'http://127.0.0.1:{api_port}')
        with (Path(directory) / 'server.log').open('w+') as log:
            process = subprocess.Popen([DOTNET, str(API)], env=env, stdout=log, stderr=log)
            try:
                for attempt in range(100):
                    if process.poll() is not None:
                        log.seek(0); raise AssertionError('API exited: ' + log.read())
                    try:
                        conn = http.client.HTTPConnection('127.0.0.1', api_port, timeout=1)
                        conn.request('GET', '/health'); response = conn.getresponse(); response.read(); conn.close()
                        break
                    except OSError:
                        time.sleep(.05)
                else:
                    raise AssertionError('API did not start')
                yield api_port
            finally:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                log.seek(0)
                if 'private-canary-do-not-log' in log.read():
                    raise AssertionError('Backend response leaked into API logs')


@unittest.skipUnless(DOTNET and API.exists(), 'Build Release API and set MAILGUARD_TEST_DOTNET first')
class ApiRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        threading.Thread(target=cls.backend.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.backend.shutdown(); cls.backend.server_close()

    def request(self, api_port, method, path, body=None, chunked=False):
        conn = http.client.HTTPConnection('127.0.0.1', api_port, timeout=5)
        try:
            if chunked:
                conn.request(method, path, [body[:100], body[100:]], headers={'Transfer-Encoding': 'chunked'}, encode_chunked=True)
            else:
                conn.request(method, path, body)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_classify_limits_and_backend_failures(self):
        Backend.mode = 'valid'
        with api_process(self.backend.server_port) as api_port:
            self.assertEqual(self.request(api_port, 'GET', '/ready')[0], 200)
            self.assertEqual(self.request(api_port, 'POST', '/classify', b'')[0], 400)
            status, body = self.request(api_port, 'POST', '/classify', b'Subject: meeting\r\n\r\nHello colleague')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['decision'], 0)
            self.assertEqual(Backend.last_type, 'message/rfc822')
            Backend.mode = 'deferred'
            status, body = self.request(api_port, 'POST', '/classify', b'Subject: hello\r\n\r\ntext')
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)['decision'], 2)  # SMTP deferral is not HAM.
            Backend.mode = 'valid'
            for chunked in (False, True):
                self.assertEqual(self.request(api_port, 'POST', '/classify', bytes(257), chunked)[0], 413)
            for mode, expected in [('invalid', 502), ('http_error', 502), ('slow', 504)]:
                Backend.mode = mode
                status, body = self.request(api_port, 'POST', '/classify', b'Subject: hello\r\n\r\ntext')
                self.assertEqual(status, expected)
                self.assertNotIn(b'private-canary', body)
        Backend.mode = 'valid'

    def test_missing_model_is_not_ready(self):
        with api_process(self.backend.server_port, model=False) as api_port:
            self.assertEqual(self.request(api_port, 'GET', '/health')[0], 200)
            self.assertEqual(self.request(api_port, 'GET', '/ready')[0], 503)
            self.assertEqual(self.request(api_port, 'POST', '/classify', b'hello')[0], 503)


if __name__ == '__main__':
    unittest.main()
