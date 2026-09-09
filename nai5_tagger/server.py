from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from nai5_tagger.compiler import assign_unassigned
from nai5_tagger.grok_vision import gateway_ready
from nai5_tagger.pipeline import PipelineError, run_pipeline
from nai5_tagger.render import render
from nai5_tagger.types import CompileOptions, Nai5Prompt, nai5_prompt_to_dict

HOST = "127.0.0.1"
PORT = 18770
URL = f"http://{HOST}:{PORT}/"
_INDEX = Path(__file__).resolve().parent / "data" / "index.html"
_GATEWAY_UNAVAILABLE = "Vision gateway unavailable. Set NAI5_TAGGER_VLM_BASE and NAI5_TAGGER_VLM_KEY."


def make_server(host: str, port: int) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _Handler)


def _port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _header_param(header: str, key: str) -> str | None:
    key_l = key.lower()
    for part in header.split(";"):
        part = part.strip()
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name.strip().lower() == key_l:
            return value.strip().strip('"')
    return None


def _parse_multipart(
    content_type: str, body: bytes
) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    boundary = _header_param(content_type, "boundary")
    if not boundary:
        return {}, {}
    token = b"--" + boundary.encode("ascii")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for raw in body.split(token):
        if not raw or raw.startswith(b"--"):
            continue
        if raw.startswith(b"\r\n"):
            raw = raw[2:]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        header_blob, sep, data = raw.partition(b"\r\n\r\n")
        if not sep:
            continue
        disposition = ""
        for line in header_blob.split(b"\r\n"):
            if b":" not in line:
                continue
            name, value = line.split(b":", 1)
            if name.decode("latin-1").strip().lower() == "content-disposition":
                disposition = value.decode("latin-1").strip()
                break
        field_name = _header_param(disposition, "name")
        if not field_name:
            continue
        filename = _header_param(disposition, "filename")
        if filename is not None:
            files[field_name] = (filename, data)
        else:
            fields[field_name] = data.decode("utf-8")
    return fields, files


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ok_payload(prompt: Nai5Prompt) -> dict:
    return {
        "prompt": nai5_prompt_to_dict(prompt),
        "text": render(prompt),
        "error": None,
    }


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                body = _INDEX.read_bytes()
            except OSError:
                self._json({"error": "index.html missing"}, 404)
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if path == "/api/ready":
            try:
                ok = bool(gateway_ready())
            except Exception:
                ok = False
            self._json({"ok": ok, "error": "" if ok else _GATEWAY_UNAVAILABLE})
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/reverse":
            self._reverse()
            return
        if path == "/api/assign":
            self._assign()
            return
        self._json({"error": "not found"}, 404)

    def _reverse(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type") or ""
        fields, files = _parse_multipart(content_type, body)
        upload = files.get("file")
        if upload is None:
            self._json({"error": "missing file"}, 400)
            return
        filename, data = upload
        suffix = Path(filename).suffix or ".png"
        include_nl = fields.get("include_nl")
        if include_nl is None:
            options = CompileOptions()
        else:
            options = CompileOptions(include_nl=_truthy(include_nl))
        path = None
        fd = -1
        try:
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.write(fd, data)
            os.close(fd)
            fd = -1
            prompt = run_pipeline(path, options)
        except PipelineError as exc:
            self._json({"error": exc.message, "exit_code": exc.exit_code})
            return
        except Exception as exc:
            self._json({"error": str(exc)})
            return
        finally:
            if fd >= 0:
                os.close(fd)
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        self._json(_ok_payload(prompt))

    def _assign(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
            prompt = Nai5Prompt.from_dict(payload["prompt"])
            result = assign_unassigned(prompt, payload["tag"], payload["dest"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json({"error": str(exc)}, 400)
            return
        self._json(_ok_payload(result))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> int:
    if _port_in_use(HOST, PORT):
        print(URL)
        return 0
    try:
        httpd = make_server(HOST, PORT)
    except OSError:
        print(URL)
        return 0
    print(URL)
    try:
        webbrowser.open(URL)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
