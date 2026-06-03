import base64
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, WebSocket, WebSocketException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBasic, HTTPBasicCredentials, HTTPBearer


basic_scheme = HTTPBasic(auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def _matches(value: Optional[str], expected: Optional[str]) -> bool:
    return bool(value and expected and secrets.compare_digest(value, expected))


def create_admin_dependency(api_key: Optional[str], username: str):
    async def _auth(
        basic: HTTPBasicCredentials = Depends(basic_scheme),
        bearer: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ):
        if not api_key:
            return
        if bearer and bearer.scheme.lower() == "bearer" and _matches(bearer.credentials, api_key):
            return
        if basic and _matches(basic.username, username) and _matches(basic.password, api_key):
            return
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Basic"},
        )

    return _auth


def authenticate_websocket(websocket: WebSocket, api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None

    auth_header = websocket.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer ") and _matches(auth_header[7:], api_key):
        return None

    for protocol in websocket.headers.get("sec-websocket-protocol", "").split(","):
        protocol = protocol.strip()
        if not protocol.startswith("Authorization."):
            continue
        try:
            b64_key = protocol[14:]
            b64_key += "=" * (-len(b64_key) % 4)
            decoded_key = base64.b64decode(b64_key).decode("utf-8")
            if _matches(decoded_key, api_key):
                return protocol
        except Exception:
            pass

    raise WebSocketException(
        code=status.WS_1008_POLICY_VIOLATION,
        reason="Invalid or missing API Key",
    )
