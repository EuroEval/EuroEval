"""Grant confidentiality, legacy ciphertext, revocation fences and POST ambiguity."""

from __future__ import annotations

import base64
import collections.abc as c
import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from euroeval_api.grants import InvalidGrant, UserGrants, WorkerIdentity

_USER = {"id": 4, "login": "volunteer"}
_HASH = hashlib.sha256(b"credential").hexdigest()
_CREDENTIAL_KEY = f"euroeval:worker:credential:{_HASH}"
_GRANT_KEY = f"euroeval:worker:user-grant:{_HASH}"


class GitHub:
    """In-memory installation/token/comment contract, including ambiguous writes."""

    def __init__(self, key: rsa.RSAPrivateKey) -> None:
        """Create an isolated provider.

        Args:
            key:
                Signing key whose public half verifies App requests.
        """
        self.key = key
        self.installation_tokens: set[str] = set()
        self.comments: list[dict[str, object]] = []
        self.repositories = [{"id": 386992479, "full_name": "EuroEval/EuroEval"}]
        self.revoke_failures = 0
        self.lost_comment = False
        self.revoked = False
        self.on_refresh: c.Callable[[], c.Awaitable[None]] | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        """Serve the exercised public GitHub contracts and their failure modes.

        Args:
            request:
                Actual HTTPX request from the public grant operation.

        Returns:
            Contract response.

        Raises:
            httpx.ReadError:
                After a committed comment loses its response.
        """
        path, method = request.url.path, request.method
        authorization = request.headers.get("authorization", "")
        body = json.loads(request.content or b"{}")
        permissions = {"issues": "write", "metadata": "read"}
        if path.startswith("/app/installations/2"):
            if not self.valid_jwt(authorization=authorization):
                return httpx.Response(401)
            if method == "GET" and path == "/app/installations/2":
                return httpx.Response(
                    200,
                    json={
                        "id": 2,
                        "account": {"login": "EuroEval"},
                        "repository_selection": "selected",
                        "permissions": permissions,
                    },
                )
            if (
                method == "POST"
                and path.endswith("/access_tokens")
                and body == {"permissions": permissions}
            ):
                self.installation_tokens.add("temporary.token")
                return httpx.Response(
                    201, json={"token": "temporary.token", "permissions": permissions}
                )
        if (
            path == "/installation/repositories"
            and authorization == "Bearer temporary.token"
            and "temporary.token" in self.installation_tokens
        ):
            return httpx.Response(200, json={"repositories": self.repositories})
        if (
            path == "/installation/token"
            and method == "DELETE"
            and authorization == "Bearer temporary.token"
        ):
            self.installation_tokens.discard("temporary.token")
            return httpx.Response(204)
        if (
            path == "/applications/client/token"
            and authorization == "Basic Y2xpZW50OnNlY3JldA=="
        ):
            if (
                method == "POST"
                and body.get("access_token") in {"access.token", "access.rotated"}
                and not self.revoked
            ):
                return httpx.Response(
                    200,
                    json={
                        "token": body["access_token"],
                        "app": {"client_id": "client", "id": 1},
                        "user": _USER,
                    },
                )
            if method == "DELETE" and body.get("access_token") in {
                "access.token",
                "access.rotated",
            }:
                if self.revoke_failures:
                    self.revoke_failures -= 1
                    return httpx.Response(503)
                self.revoked = True
                return httpx.Response(204)
        if (
            path == "/login/oauth/access_token"
            and method == "POST"
            and body.get("client_id") == "client"
            and body.get("client_secret") == "secret"
            and body.get("refresh_token") == "refresh.token"
        ):
            if self.on_refresh is not None:
                await self.on_refresh()
            return httpx.Response(
                200,
                json={
                    "access_token": "access.rotated",
                    "refresh_token": "refresh.rotated",
                    "expires_in": 3600,
                    "refresh_token_expires_in": 86400,
                },
            )
        if (
            authorization in {"Bearer access.token", "Bearer access.rotated"}
            and not self.revoked
        ):
            if path == "/user":
                return httpx.Response(200, json=_USER)
            if path == "/repos/EuroEval/EuroEval/issues/7/comments":
                if method == "GET":
                    page = int(request.url.params.get("page", "1"))
                    return httpx.Response(
                        200, json=self.comments[(page - 1) * 100 : page * 100]
                    )
                if method == "POST":
                    comment = {"body": body["body"], "user": _USER}
                    self.comments.append(comment)
                    if self.lost_comment:
                        self.lost_comment = False
                        raise httpx.ReadError(
                            "Committed response lost", request=request
                        )
                    return httpx.Response(201, json=comment)
        return httpx.Response(401)

    def valid_jwt(self, authorization: str) -> bool:
        """Verify the provider-facing RS256 assertion, rather than accepting a stub.

        Args:
            authorization:
                App bearer authorization.

        Returns:
            Whether signature, algorithm, issuer, and lifetime are valid.
        """
        try:
            header, payload, signature = authorization.removeprefix("Bearer ").split(
                "."
            )
            decoded = json.loads(
                base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            )
            self.key.public_key().verify(
                base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)),
                f"{header}.{payload}".encode(),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            algorithm = json.loads(
                base64.urlsafe_b64decode(header + "=" * (-len(header) % 4))
            )
            return (
                algorithm["alg"] == "RS256"
                and decoded["iss"] == "1"
                and decoded["iat"] <= time.time() < decoded["exp"] <= time.time() + 600
            )
        except (ValueError, InvalidSignature, KeyError):
            return False


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> GitHub:
    """Configure a fake App with real cryptographic keys and no live credentials.

    Args:
        monkeypatch:
            Isolated environment editor.

    Returns:
        In-memory provider.
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    values = {
        "VOLUNTEER_GITHUB_APP_CLIENT_ID": "client",
        "VOLUNTEER_GITHUB_APP_CLIENT_SECRET": "secret",
        "VOLUNTEER_GITHUB_APP_ID": "1",
        "VOLUNTEER_GITHUB_APP_PRIVATE_KEY": key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode(),
        "VOLUNTEER_GITHUB_APP_INSTALLATION_ID": "2",
        "VOLUNTEER_GITHUB_APP_REPOSITORY_ID": "386992479",
        "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY": base64.b64encode(
            bytes(range(32))
        ).decode(),
    }
    for name, value in values.items():
        monkeypatch.setenv(name=name, value=value)
    return GitHub(key=key)


@pytest.fixture
async def grants(github: GitHub, upstash_client: Redis) -> c.AsyncIterator[UserGrants]:
    """Provide the public manager using real Redis and provider HTTP contracts.

    Args:
        github:
            Isolated GitHub provider.
        upstash_client:
            Official SDK connected to real Redis.

    Yields:
        Public grant manager.
    """
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler=github)
    ) as client:
        yield UserGrants(http=client, redis=upstash_client)


async def owned() -> None:
    """Supply an issue fence for grant-only outcome tests."""


@pytest.mark.anyio
async def test_ambiguous_comment_response_never_repeats_post(
    grants: UserGrants, github: GitHub, upstash_client: Redis
) -> None:
    """Reconcile a committed-but-lost response and refuse revoked user authority.

    Args:
        grants:
            Public grant manager.
        github:
            Provider with an ambiguous comment response.
        upstash_client:
            Real Redis SDK.
    """
    identity = await retain(grants=grants, redis=upstash_client)
    github.lost_comment = True
    with pytest.raises(httpx.ReadError):
        await grants.ensure_volunteer_comment(
            identity=identity, issue_number=7, assert_owned=owned
        )
    await grants.ensure_volunteer_comment(
        identity=identity, issue_number=7, assert_owned=owned
    )
    assert len(github.comments) == 1
    await grants.revoke_user_grant(identity=identity)
    with pytest.raises(InvalidGrant):
        await grants.ensure_volunteer_comment(
            identity=identity, issue_number=7, assert_owned=owned
        )
    assert len(github.comments) == 1


async def retain(
    grants: UserGrants, redis: Redis, access_seconds: int = 3600
) -> WorkerIdentity:
    """Issue the credential prerequisite and retain authority via the public method.

    Args:
        grants:
            Public grant manager.
        redis:
            Official SDK client.
        access_seconds (optional):
            Access lifetime, allowing immediate rotation tests.

    Returns:
        Verified identity.
    """
    await redis.set(
        key=_CREDENTIAL_KEY,
        value=json.dumps({"contributor": "volunteer", "grant_mode": "app"}),
        ex=3600,
    )
    now = int(time.time() * 1000)
    await grants.retain_user_grant(
        access_token="access.token",
        refresh_token="refresh.token",
        access_expires=now + access_seconds * 1000,
        refresh_expires=now + 86400000,
        credential="credential",
        expected_login="volunteer",
    )
    return {"hash": _HASH, "contributor": "volunteer"}


@pytest.mark.anyio
async def test_config_loss_still_revokes_local_authority(
    grants: UserGrants,
    github: GitHub,
    upstash_client: Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain encrypted cleanup when configuration disappears during revocation.

    Args:
        grants:
            Public grant manager.
        github:
            Provider's observable revocation state.
        upstash_client:
            Real Redis SDK.
        monkeypatch:
            Simulated partial configuration loss.
    """
    identity = await retain(grants=grants, redis=upstash_client)
    monkeypatch.delenv("VOLUNTEER_GITHUB_APP_ID")
    await grants.revoke_user_grant(identity=identity)
    assert await upstash_client.get(_CREDENTIAL_KEY) is None
    assert await upstash_client.get(_GRANT_KEY) is None
    assert isinstance(await upstash_client.get(f"{_GRANT_KEY}:revocation"), str)
    assert not github.revoked
    monkeypatch.setenv("VOLUNTEER_GITHUB_APP_ID", "1")
    await grants.revoke_user_grant(identity=identity)
    assert github.revoked


@pytest.mark.anyio
async def test_deployed_typescript_envelope_remains_usable(
    grants: UserGrants, github: GitHub, upstash_client: Redis
) -> None:
    """Read an independently generated WebCrypto envelope and post as its user.

    Args:
        grants:
            Public grant manager.
        github:
            Provider enforcing the decrypted user's access token.
        upstash_client:
            Real Redis SDK.
    """
    vector = json.loads(
        (Path(__file__).parent / "fixtures/typescript-user-grant.json").read_text()
    )
    identity: WorkerIdentity = {"hash": vector["hash"], "contributor": "volunteer"}
    await upstash_client.set(
        key=_CREDENTIAL_KEY,
        value=json.dumps({"contributor": "volunteer", "grant_mode": "app"}),
        ex=3600,
    )
    await upstash_client.set(
        key=_GRANT_KEY, value=json.dumps(vector["envelope"]), ex=3600
    )
    await grants.ensure_volunteer_comment(
        identity=identity, issue_number=7, assert_owned=owned
    )
    assert len(github.comments) == 1
    assert github.comments[0]["user"] == _USER


@pytest.mark.anyio
async def test_encrypted_retention_and_retryable_revoke(
    grants: UserGrants, github: GitHub, upstash_client: Redis
) -> None:
    """Keep encrypted retry state while removing local authority before remote retry.

    Args:
        grants:
            Public grant manager.
        github:
            Provider failure controls and observable remote state.
        upstash_client:
            Real Redis SDK.
    """
    identity = await retain(grants=grants, redis=upstash_client)
    encrypted = await upstash_client.get(_GRANT_KEY)
    assert isinstance(encrypted, str)
    assert "access.token" not in encrypted and "refresh.token" not in encrypted
    assert not github.installation_tokens
    github.revoke_failures = 1
    with pytest.raises(HTTPException) as failure:
        await grants.revoke_user_grant(identity=identity)
    assert failure.value.status_code == 503
    assert await upstash_client.get(_CREDENTIAL_KEY) is None
    assert await upstash_client.get(_GRANT_KEY) is None
    assert isinstance(await upstash_client.get(f"{_GRANT_KEY}:revocation"), str)
    assert not github.revoked
    await grants.revoke_user_grant(identity=identity)
    assert github.revoked
    assert await upstash_client.get(f"{_GRANT_KEY}:revocation") is None


@pytest.mark.anyio
async def test_invalid_repository_still_revokes_temporary_token(
    grants: UserGrants, github: GitHub, upstash_client: Redis
) -> None:
    """Reject expanded installation scope without retaining temporary authority.

    Args:
        grants:
            Public grant manager.
        github:
            Provider exposing an incorrectly selected repository.
        upstash_client:
            Real Redis SDK.
    """
    github.repositories.append({"id": 99, "full_name": "EuroEval/another"})
    with pytest.raises(HTTPException) as failure:
        await retain(grants=grants, redis=upstash_client)
    assert failure.value.status_code == 503
    assert await upstash_client.get(_GRANT_KEY) is None
    assert not github.installation_tokens


@pytest.mark.anyio
async def test_rotation_cannot_restore_revoked_authority(
    grants: UserGrants, github: GitHub, upstash_client: Redis
) -> None:
    """A credential expiring during OAuth rotation prevents the new envelope commit.

    Args:
        grants:
            Public grant manager.
        github:
            Provider with a controllable in-flight rotation.
        upstash_client:
            Real Redis SDK.
    """
    identity = await retain(grants=grants, redis=upstash_client, access_seconds=30)
    encrypted = await upstash_client.get(_GRANT_KEY)

    async def expire_credential() -> None:
        """Expire authority while the remote rotation is in flight."""
        await upstash_client.delete(_CREDENTIAL_KEY)

    github.on_refresh = expire_credential
    with pytest.raises(InvalidGrant):
        await grants.ensure_volunteer_comment(
            identity=identity, issue_number=7, assert_owned=owned
        )
    assert await upstash_client.get(_CREDENTIAL_KEY) is None
    assert await upstash_client.get(_GRANT_KEY) == encrypted
    assert not github.comments
