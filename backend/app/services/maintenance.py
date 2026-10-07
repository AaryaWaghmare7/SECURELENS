"""Run retention cleanup in the API instance that owns the private storage disk."""
import asyncio
import logging

from starlette.concurrency import run_in_threadpool

from .records import cleanup_expired

LOGGER = logging.getLogger(__name__)


def cleanup_once(app):
    with app.state.session_factory() as session:
        cleanup_expired(session, app.state.storage)


async def retention_worker(app, stopped):
    while not stopped.is_set():
        try:
            await run_in_threadpool(cleanup_once, app)
        except Exception as error:
            # Database exceptions can contain personal SQL parameters; don't log them.
            LOGGER.error("Preview retention cleanup failed (%s).", type(error).__name__)
        try:
            await asyncio.wait_for(stopped.wait(), timeout=app.state.settings.storage_cleanup_interval_seconds)
        except TimeoutError:
            pass
