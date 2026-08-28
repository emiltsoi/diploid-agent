"""Dummy ingress handler used by test_ingress."""

from __future__ import annotations

from fastapi import Request, Response

from diploid_agent.transport.ingress import IngressHandler


class Ingress(IngressHandler):
    async def handle(self, request: Request) -> Response:
        body = await request.body()
        return Response(f"mesh-ok:{body.decode()}", media_type="text/plain", status_code=202)
