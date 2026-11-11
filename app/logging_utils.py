"""
Sistem de logging structurat pentru cercetare.

Scop: a putea reconstitui, pentru orice cerere a utilizatorului, lantul complet
de functii/servicii apelate (web -> LLM -> tool MCP -> API extern), cu durata
fiecarei etape in milisecunde.

Mecanism:
- Fiecare cerere primeste un `request_id` scurt, propagat:
    * intre etape din acelasi proces prin contextvars (functioneaza corect
      peste `await`/`asyncio.gather`);
    * intre procese (web_server -> mcp_server / mcp_server_ors) prin
      header-ul HTTP `X-Request-Id`.
- Fiecare etapa relevanta e masurata cu `log_timing(...)`, care scrie un log
  la inceput si unul la sfarsit (cu `duration_ms` si `status`).
- Logurile sunt scrise simultan:
    * in consola, format text colorat, usor de citit live;
    * intr-un fisier JSON Lines (`logs/<service>.log`, rotativ), usor de
      incarcat in pandas/Excel pentru analiza statistica.
"""
import contextvars
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

REQUEST_ID_HEADER = "X-Request-Id"

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_request_id_var: "contextvars.ContextVar[str]" = contextvars.ContextVar("request_id", default="-")

# Loggere de biblioteci terte care sunt foarte vorbarete la nivel INFO si
# ascund semnal util (httpx loghează fiecare request, openai la fel).
_NOISY_LOGGERS = ("httpx", "httpcore", "openai")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:8]
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_var.get()


def set_request_id(rid: str) -> None:
    _request_id_var.set(rid)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        metric = getattr(record, "metric", None)
        if metric:
            payload.update(metric)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\x1b[90m", "INFO": "\x1b[36m", "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m", "CRITICAL": "\x1b[41m",
    }
    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"

    def __init__(self):
        super().__init__()
        self._use_color = sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        rid = getattr(record, "request_id", "-")
        line = f"{ts} | {record.levelname:<8} | {rid:<8} | {record.name:<10} | {record.getMessage()}"
        metric = getattr(record, "metric", None)
        if metric and "duration_ms" in metric:
            line += f"  ({metric['duration_ms']} ms)"
        if not self._use_color:
            return line
        color = self._COLORS.get(record.levelname, "")
        return f"{color}{line}{self._RESET}"


def setup_logging(service_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configureaza logging-ul global al procesului curent.

    Scrie simultan in consola (text) si intr-un fisier JSON Lines rotativ
    (logs/<service_name>.log). Trebuie apelata o singura data, la pornirea
    fiecarui server.
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ConsoleFormatter())
    console.addFilter(_RequestIdFilter())
    root.addHandler(console)

    file_path = os.path.join(LOG_DIR, f"{service_name}.log")
    file_handler = RotatingFileHandler(file_path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    file_handler.addFilter(_RequestIdFilter())
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger(service_name)


@contextmanager
def log_timing(logger: logging.Logger, stage: str, **fields):
    """Masoara si logheaza durata unei etape (functioneaza si in jurul unui `await`).

    Exemplu:
        with log_timing(log, "llm.completion", round=1, model="gpt-4o-mini"):
            resp = await openai.chat.completions.create(...)
    """
    start = time.perf_counter()
    logger.info(f"-> {stage}", extra={"metric": {"stage": stage, "event": "start", **fields}})
    try:
        yield
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.error(
            f"<- {stage} FAILED: {exc}",
            extra={"metric": {"stage": stage, "event": "end", "duration_ms": duration_ms,
                               "status": "error", "error": str(exc), **fields}},
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            f"<- {stage} ok",
            extra={"metric": {"stage": stage, "event": "end", "duration_ms": duration_ms,
                               "status": "ok", **fields}},
        )


try:
    from starlette.middleware.base import BaseHTTPMiddleware

    class RequestIdMiddleware(BaseHTTPMiddleware):
        """Propaga/genereaza request_id pe fiecare cerere HTTP si logheaza
        automat inceputul/sfarsitul + durata, pentru orice endpoint FastAPI.
        """

        def __init__(self, app, logger_name: str = "http"):
            super().__init__(app)
            self._log = logging.getLogger(logger_name)

        async def dispatch(self, request, call_next):
            # Prioritate: (1) header explicit, (2) request_id deja activ in
            # contextul curent (apel ASGI imbricat in acelasi task asyncio,
            # ex: FastMCP care rescrie cererea catre propriul transport HTTP
            # fara sa propage header-ul), (3) request nou la nivel de top -> id nou.
            incoming = request.headers.get(REQUEST_ID_HEADER)
            if incoming:
                rid = incoming
            elif get_request_id() != "-":
                rid = get_request_id()
            else:
                rid = new_request_id()
            set_request_id(rid)
            stage = f"http.{request.url.path}"
            start = time.perf_counter()

            # Pentru /mcp: extrage method/tool/args din body-ul JSON-RPC.
            # Starlette cacheaza body-ul citit aici si il repeta catre
            # handler-ul real, deci request-ul efectiv nu e afectat.
            rpc_info = {}
            if request.method == "POST" and request.url.path.endswith("/mcp"):
                try:
                    payload = json.loads(await request.body())
                    rpc_method = payload.get("method")
                    if rpc_method:
                        rpc_info["rpc_method"] = rpc_method
                    if rpc_method == "tools/call":
                        params = payload.get("params", {})
                        rpc_info["tool"] = params.get("name")
                        rpc_info["tool_args"] = params.get("arguments")
                except Exception:
                    pass

            self._log.info(
                f"-> {request.method} {request.url.path}"
                + (f" [{rpc_info.get('rpc_method')}] {rpc_info.get('tool', '')} {rpc_info.get('tool_args', '')}"
                   if rpc_info.get("rpc_method") else ""),
                extra={"metric": {"stage": stage, "event": "start", "method": request.method, **rpc_info}},
            )
            try:
                response = await call_next(request)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - start) * 1000, 1)
                self._log.error(
                    f"<- {request.method} {request.url.path} FAILED: {exc}",
                    extra={"metric": {"stage": stage, "event": "end", "duration_ms": duration_ms,
                                       "status": "error", "error": str(exc)}},
                )
                raise
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            response.headers[REQUEST_ID_HEADER] = rid
            self._log.info(
                f"<- {request.method} {request.url.path} {response.status_code}",
                extra={"metric": {"stage": stage, "event": "end", "duration_ms": duration_ms,
                                   "status": "ok", "status_code": response.status_code}},
            )
            return response

except ImportError:  # starlette indisponibil (nu ar trebui sa fie cazul)
    RequestIdMiddleware = None  # type: ignore
