from collections import deque
from time import monotonic

from starlette.exceptions import HTTPException


class RequestSizeLimit:
    """Bound streamed bodies too, including uploads without Content-Length."""

    def __init__(self, app, settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.settings.max_request_bytes:
                    raise HTTPException(413, "The request exceeds the upload limit.")
            return message

        await self.app(scope, bounded_receive, send)


class AuthRateLimit:
    """Per-process throttling; a shared limiter is required for multiple workers."""

    def __init__(self, maximum):
        self.maximum = maximum
        self.requests = {}

    def allows(self, address):
        now = monotonic()
        # Expire idle keys so arbitrary client addresses cannot grow memory forever.
        for key in list(self.requests):
            queue = self.requests[key]
            while queue and queue[0] <= now - 60:
                queue.popleft()
            if not queue:
                del self.requests[key]
        queue = self.requests.setdefault(address, deque())
        if len(queue) >= self.maximum:
            return False
        queue.append(now)
        return True
