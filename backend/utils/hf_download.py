"""Hugging Face download retry helpers.

Voicebox normally downloads from the official Hugging Face endpoint. Some
networks terminate that TLS connection before the Hub API responds, while a
compatible mirror remains reachable. A model download may retry once through
the mirror only for connectivity failures; authentication, repository, and
model-loading errors must keep their original meaning.
"""

import asyncio
import inspect
import logging
import os
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
DEFAULT_HF_FALLBACK_ENDPOINT = "https://hf-mirror.com"

def get_current_hf_endpoint() -> str:
    """Return the endpoint currently used by the loaded HF client."""
    try:
        from huggingface_hub import constants as hf_constants

        return hf_constants.ENDPOINT.rstrip("/")
    except ImportError:
        return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")


def get_fallback_hf_endpoint() -> str | None:
    """Return the configured fallback, or ``None`` when explicitly disabled."""
    configured = os.environ.get(
        "VOICEBOX_HF_FALLBACK_ENDPOINT",
        DEFAULT_HF_FALLBACK_ENDPOINT,
    ).strip()
    if configured.lower() in ("", "0", "false", "off", "none"):
        return None
    return configured.rstrip("/")


def _exception_text(error: BaseException) -> str:
    """Flatten nested request exceptions without depending on requests/httpx."""
    parts: list[str] = []
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)
        for arg in current.args:
            if isinstance(arg, BaseException):
                pending.append(arg)
    return " ".join(parts).lower()


def is_hf_connectivity_error(error: BaseException) -> bool:
    """Identify failures for which changing the download endpoint can help."""
    if isinstance(error, (ConnectionError, TimeoutError, ssl.SSLError, asyncio.TimeoutError)):
        return True

    text = _exception_text(error)
    connectivity_markers = (
        "connection aborted",
        "connection error",
        "connection refused",
        "connection reset",
        "max retries exceeded",
        "name resolution",
        "network is unreachable",
        "proxyerror",
        "read timed out",
        "remote disconnected",
        "sslerror",
        "ssleoferror",
        "temporary failure in name resolution",
        "timed out",
        "unexpected_eof_while_reading",
    )
    return any(marker in text for marker in connectivity_markers)


def activate_hf_endpoint(endpoint: str) -> None:
    """Switch new Hugging Face requests to *endpoint* in the live process."""
    normalized = endpoint.rstrip("/")
    os.environ["HF_ENDPOINT"] = normalized

    from huggingface_hub import constants as hf_constants

    # huggingface_hub caches HF_ENDPOINT at import time, so updating the
    # environment alone would leave this already-running desktop server on
    # the broken URL.
    hf_constants.ENDPOINT = normalized

    try:
        from huggingface_hub.utils._http import reset_sessions

        reset_sessions()
    except (ImportError, AttributeError):
        # Older bundled huggingface_hub versions do not expose reset_sessions;
        # newly created clients still read the updated constants.ENDPOINT.
        pass

    logger.warning("Hugging Face endpoint switched to fallback: %s", normalized)


async def _invoke(load_func: Callable[[], Any | Awaitable[Any]]) -> Any:
    result = load_func()
    if inspect.isawaitable(result):
        return await result
    return result


async def run_with_hf_fallback(
    load_func: Callable[[], Any | Awaitable[Any]],
    on_retry: Callable[[str], None] | None = None,
) -> str | None:
    """Run a model load and retry once on the fallback endpoint if warranted.

    Returns the fallback endpoint when a retry was used, otherwise ``None``.
    The fallback is deliberately limited to connectivity failures from the
    official endpoint so access-control and invalid-model errors are never
    obscured by a second request to another service.
    """
    try:
        await _invoke(load_func)
        return None
    except Exception as error:
        current_endpoint = get_current_hf_endpoint()
        fallback_endpoint = get_fallback_hf_endpoint()
        if (
            not fallback_endpoint
            or current_endpoint != DEFAULT_HF_ENDPOINT
            or fallback_endpoint == current_endpoint
            or not is_hf_connectivity_error(error)
        ):
            raise

        logger.warning(
            "Official Hugging Face endpoint failed (%s); retrying via %s",
            error,
            fallback_endpoint,
        )
        activate_hf_endpoint(fallback_endpoint)
        if on_retry:
            on_retry(fallback_endpoint)
        await _invoke(load_func)
        return fallback_endpoint
