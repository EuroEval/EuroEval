"""Encrypted GitHub App authority, fenced rotation, and public comment deduplication."""

from __future__ import annotations

import asyncio
import base64
import collections.abc as c
import hashlib
import json
import math
import os
import re
import secrets
import time
import typing as t
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException
from upstash_redis.asyncio import Redis

from .dependencies import fetch_with_retry, require_env
from .state import redis_get_json

_NAMES = (
    "VOLUNTEER_GITHUB_APP_CLIENT_ID",
    "VOLUNTEER_GITHUB_APP_CLIENT_SECRET",
    "VOLUNTEER_GITHUB_APP_ID",
    "VOLUNTEER_GITHUB_APP_PRIVATE_KEY",
    "VOLUNTEER_GITHUB_APP_INSTALLATION_ID",
    "VOLUNTEER_GITHUB_APP_REPOSITORY_ID",
    "VOLUNTEER_GITHUB_APP_TOKEN_ENCRYPTION_KEY",
)
_PREFIX = "euroeval:worker:user-grant:"
_CREDENTIAL = "euroeval:worker:credential:"
_PERMISSIONS = {"issues": "write", "metadata": "read"}
_COMMENT = "I'd like to evaluate this."
_RENEW = (
    "if redis.call('GET',KEYS[1]) == ARGV[1] then return "
    "redis.call('EXPIRE',KEYS[1],ARGV[2]) else return 0 end"
)
_DELETE = (
    "if redis.call('GET',KEYS[1]) == ARGV[1] then return "
    "redis.call('DEL',KEYS[1]) else return 0 end"
)
_PERSIST = """
if redis.call('GET',KEYS[1]) ~= ARGV[1] then return 0 end;
if redis.call('EXISTS',KEYS[2]) == 0 then return -1 end;
local old=redis.call('GET',KEYS[3]);
if ARGV[2] ~= '' and (not old or cjson.decode(old).generation ~= ARGV[2])
then return -2 end;
redis.call('SET',KEYS[3],ARGV[3],'EX',ARGV[4]); return 1
"""
_REMOVE = """
if redis.call('GET',KEYS[1]) ~= ARGV[1] then return 0 end;
local old=redis.call('GET',KEYS[2]);
if ARGV[2] ~= '' and old ~= ARGV[2] then return -1 end;
redis.call('DEL',KEYS[2],KEYS[3]); return 1
"""

_CLEAR_PENDING = """
if redis.call('GET',KEYS[1]) == ARGV[1] then
return redis.call('DEL',KEYS[2]) else return 0 end
"""


class InvalidGrant(HTTPException):
    """Renewable authority is missing, revoked, corrupted, or from another identity."""

    def __init__(self) -> None:
        """Build the deployed reauthorization response."""
        super().__init__(
            status_code=401,
            detail={
                "error": "GitHub App authorization must be renewed.",
                "code": "github_grant_invalid",
            },
        )


@dataclass(frozen=True)
class _Config:
    """Concrete installation identity and non-printable encryption credentials."""

    client_id: str
    client_secret: str = field(repr=False)
    app_id: str
    private_key: str = field(repr=False)
    installation_id: str
    repository_id: str
    key: bytes = field(repr=False)


def _decode(value: str) -> bytes:
    """Decode the URL-safe/unpadded and standard base64 forms used by old grants.

    Args:
        value:
            Encoded configuration or envelope bytes.

    Returns:
        Decoded bytes.
    """
    compact = re.sub(r"[\t\r\n\f ]", "", value)
    return base64.b64decode(
        compact + "=" * (-len(compact) % 4), altchars=b"-_", validate=True
    )


def _encode(value: bytes) -> str:
    """Encode the deployed URL-safe unpadded base64 form.

    Args:
        value:
            Binary envelope or JWT data.

    Returns:
        Encoded text.
    """
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def app_grant_enabled() -> bool:
    """Enable App grants only for a complete, valid installation configuration.

    Returns:
        Whether App grants are configured.

    Raises:
        HTTPException:
            If partial or invalid configuration would weaken authority.
    """
    present = [name for name in _NAMES if os.getenv(name)]
    if not present:
        return False
    if len(present) != len(_NAMES):
        raise HTTPException(503, "GitHub App grant configuration is incomplete.")
    if os.getenv(_NAMES[5]) != "386992479":
        raise HTTPException(503, "GitHub App repository configuration is invalid.")
    try:
        key = _decode(value=require_env(name=_NAMES[6]))
    except ValueError as error:
        raise HTTPException(
            503, "GitHub App token encryption key must be 32 bytes in base64."
        ) from error
    if len(key) != 32:
        raise HTTPException(
            503, "GitHub App token encryption key must be 32 bytes in base64."
        )
    return True


class WorkerIdentity(t.TypedDict):
    """A verified contributor and the hash of their opaque broker credential."""

    hash: str
    contributor: str


class _Grant(t.TypedDict):
    """The persisted, encrypted grant wire format."""

    version: int
    generation: str
    userId: int
    contributor: str
    credentialHash: str
    clientId: str
    access: str
    refresh: str
    accessExpires: int
    refreshExpires: int


class UserGrants:
    """Retain minimal GitHub authority without exposing or resurrecting user tokens."""

    def __init__(self, http: httpx.AsyncClient, redis: Redis) -> None:
        """Use the application's existing connection pools.

        Args:
            http:
                Outbound HTTP client.
            redis:
                Official asynchronous Redis client.
        """
        self.http = http
        self.redis = redis

    async def discard_user_access_token(self, access_token: str) -> None:
        """Revoke a partially verified onboarding token before discarding it.

        Args:
            access_token:
                App user access token.

        Raises:
            HTTPException:
                If remote cleanup remains unconfirmed.
        """
        config = self._config()
        response = await self._request(
            url=f"https://api.github.com/applications/{config.client_id}/token",
            method="DELETE",
            authorization=self._basic(config=config),
            body={"access_token": access_token},
        )
        if not response.is_success and response.status_code != 404:
            raise HTTPException(503, "GitHub App token cleanup failed.")

    def _basic(self, config: _Config) -> str:
        """Create the App's OAuth token-management authorization.

        Args:
            config:
                Exact App configuration.

        Returns:
            Basic authorization header.
        """
        return (
            "Basic "
            + base64.b64encode(
                f"{config.client_id}:{config.client_secret}".encode()
            ).decode()
        )

    def _config(self) -> _Config:
        """Resolve exact installation identity and encryption configuration.

        Returns:
            Validated concrete configuration.

        Raises:
            HTTPException:
                If App grants are not configured.
        """
        if not app_grant_enabled():
            raise HTTPException(503, "GitHub App grants are not configured.")
        values = [require_env(name=name) for name in _NAMES]
        return _Config(
            client_id=values[0],
            client_secret=values[1],
            app_id=values[2],
            private_key=values[3].replace("\\n", "\n"),
            installation_id=values[4],
            repository_id=values[5],
            key=_decode(value=values[6]),
        )

    async def _request(
        self,
        url: str,
        method: str = "GET",
        *,
        authorization: str = "",
        body: object | None = None,
        accept: str = "application/vnd.github+json",
    ) -> httpx.Response:
        """Retry safe reads without blindly replaying a credential mutation.

        Args:
            url:
                GitHub endpoint.
            method (optional):
                HTTP method.
            authorization (optional):
                Required authorization header.
            body (optional):
                JSON request body.
            accept (optional):
                Expected response representation.

        Returns:
            Provider response.
        """
        headers = {"accept": accept, "x-github-api-version": "2022-11-28"}
        if authorization:
            headers["authorization"] = authorization
        return await fetch_with_retry(
            http=self.http, method=method, url=url, headers=headers, body=body
        )

    async def ensure_volunteer_comment(
        self,
        identity: WorkerIdentity,
        issue_number: int,
        assert_owned: c.Callable[[], c.Awaitable[None]],
    ) -> None:
        """Post once as the verified claimant, preserving ambiguous-response dedup.

        Args:
            identity:
                Authenticated claimant.
            issue_number:
                Evaluation request.
            assert_owned:
                Issue-ownership fence required immediately before mutation.

        Raises:
            InvalidGrant:
                If authority was revoked or changed.
            HTTPException:
                If history, locks, or the provider's author confirmation fail.
        """
        config = self._config()
        refreshed = await self._live(identity=identity, config=config)
        key = f"{_PREFIX}{identity['hash']}"
        async with self._lock(key=f"{key}:lock"):
            encrypted = await self.redis.get(key)
            if not isinstance(encrypted, str):
                raise InvalidGrant()
            grant = self._unseal(
                raw=encrypted, config=config, credential_hash=identity["hash"]
            )
            if (
                grant["contributor"].lower() != identity["contributor"].lower()
                or grant["generation"] != refreshed["generation"]
                or grant["accessExpires"] <= time.time() * 1000
            ):
                raise InvalidGrant()
            comment_key = f"{key}:comment:{issue_number}"
            lock_key = f"{comment_key}:lock"
            async with self._lock(key=lock_key) as comment_token:
                pending = await self.redis.get(comment_key)
                comments = await self._comments(
                    issue_number=issue_number, access=grant["access"]
                )
                ttl = max(
                    60, math.ceil((grant["refreshExpires"] - time.time() * 1000) / 1000)
                )
                for comment in comments:
                    user = comment.get("user")
                    if (
                        isinstance(user, dict)
                        and user.get("id") == grant["userId"]
                        and user.get("login", "").lower()
                        == grant["contributor"].lower()
                        and comment.get("body") == _COMMENT
                    ):
                        await self.redis.set(key=comment_key, value="done", ex=ttl)
                        return
                if pending:
                    return
                await assert_owned()
                credential = await redis_get_json(
                    redis=self.redis, key=f"{_CREDENTIAL}{identity['hash']}"
                )
                latest = await self.redis.get(key)
                if (
                    not credential
                    or not isinstance(latest, str)
                    or json.loads(latest)["generation"] != grant["generation"]
                ):
                    raise InvalidGrant()
                await self.redis.set(key=comment_key, value="pending", ex=ttl)
                post_started = False

                async def clear_pending() -> None:
                    """Clear only this comment lock's unstarted or rejected POST."""
                    with suppress(Exception):
                        await self.redis.execute(
                            command=[
                                "EVAL",
                                (_CLEAR_PENDING),
                                "2",
                                lock_key,
                                comment_key,
                                comment_token,
                            ]
                        )

                try:
                    await assert_owned()
                    latest = await self.redis.get(key)
                    if (
                        await self.redis.get(lock_key) != comment_token
                        or not isinstance(latest, str)
                        or json.loads(latest)["generation"] != grant["generation"]
                        or not await redis_get_json(
                            redis=self.redis, key=f"{_CREDENTIAL}{identity['hash']}"
                        )
                    ):
                        raise InvalidGrant()
                    await assert_owned()
                    if await self.redis.get(lock_key) != comment_token:
                        raise HTTPException(
                            409, "Volunteer comment ownership fence was lost."
                        )
                    post_started = True
                    posted = await self._request(
                        url=f"https://api.github.com/repos/EuroEval/EuroEval/issues/{issue_number}/comments",
                        method="POST",
                        authorization=f"Bearer {grant['access']}",
                        body={"body": _COMMENT},
                    )
                    if posted.status_code == 401:
                        await clear_pending()
                        raise InvalidGrant()
                    if not posted.is_success:
                        await clear_pending()
                        raise HTTPException(
                            502, "Unable to post volunteer issue comment."
                        )
                    user = posted.json().get("user") or {}
                    if (
                        user.get("id") != grant["userId"]
                        or user.get("login", "").lower() != grant["contributor"].lower()
                    ):
                        raise HTTPException(
                            502, "GitHub did not confirm the volunteer comment author."
                        )
                    await self.redis.set(key=comment_key, value="done", ex=ttl)
                except Exception:
                    if not post_started:
                        await clear_pending()
                    raise

    async def _comments(
        self, issue_number: int, access: str
    ) -> list[dict[str, object]]:
        """Scan the complete bounded comment history using the user's authority.

        Args:
            issue_number:
                Evaluation request.
            access:
                Verified user token.

        Returns:
            Complete comment history.

        Raises:
            InvalidGrant:
                If the user token is no longer authorized.
            HTTPException:
                If a complete history cannot be verified.
        """
        comments: list[dict[str, object]] = []
        for page in range(1, 101):
            response = await self._request(
                url=(
                    "https://api.github.com/repos/EuroEval/EuroEval/issues/"
                    f"{issue_number}/comments?per_page=100&page={page}"
                ),
                authorization=f"Bearer {access}",
            )
            if response.status_code == 401:
                raise InvalidGrant()
            if not response.is_success:
                raise HTTPException(502, "Unable to verify volunteer issue comments.")
            batch = response.json()
            comments.extend(batch)
            if len(batch) < 100:
                return comments
        raise HTTPException(502, "Issue comment history is too large to verify safely.")

    async def _live(self, identity: WorkerIdentity, config: _Config) -> _Grant:
        """Rotate expiring access under the same generation and credential fences.

        Args:
            identity:
                Authenticated broker identity.
            config:
                Current App configuration.

        Returns:
            Current authenticated access record.

        Raises:
            InvalidGrant:
                If rotation, identity, or persisted authority is invalid.
        """
        key = f"{_PREFIX}{identity['hash']}"
        async with self._lock(key=f"{key}:lock") as token:
            encrypted = await self.redis.get(key)
            if not isinstance(encrypted, str):
                raise InvalidGrant()
            grant = self._unseal(
                raw=encrypted, config=config, credential_hash=identity["hash"]
            )
            if (
                grant["contributor"].lower() != identity["contributor"].lower()
                or grant["refreshExpires"] <= time.time() * 1000
            ):
                raise InvalidGrant()
            if grant["accessExpires"] > time.time() * 1000 + 60000:
                return grant
            response = await self._request(
                url="https://github.com/login/oauth/access_token",
                method="POST",
                accept="application/json",
                body={
                    "client_id": config.client_id,
                    "client_secret": config.client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": grant["refresh"],
                },
            )
            rotated = response.json()
            if (
                not response.is_success
                or rotated.get("error")
                or not all(
                    rotated.get(name)
                    for name in (
                        "access_token",
                        "refresh_token",
                        "expires_in",
                        "refresh_token_expires_in",
                    )
                )
            ):
                raise InvalidGrant()
            await self._verify_user(
                config=config, access=rotated["access_token"], expected=grant
            )
            now = int(time.time() * 1000)
            next_record: _Grant = {
                **grant,
                "generation": secrets.token_urlsafe(nbytes=18),
                "access": rotated["access_token"],
                "refresh": rotated["refresh_token"],
                "accessExpires": now + rotated["expires_in"] * 1000,
                "refreshExpires": now + rotated["refresh_token_expires_in"] * 1000,
            }
            ttl = min(2592000, math.ceil((next_record["refreshExpires"] - now) / 1000))
            outcome = await self.redis.execute(
                command=[
                    "EVAL",
                    _PERSIST,
                    "3",
                    f"{key}:lock",
                    f"{_CREDENTIAL}{identity['hash']}",
                    key,
                    token,
                    grant["generation"],
                    self._seal(record=next_record, config=config),
                    str(ttl),
                ]
            )
            if outcome != 1:
                raise InvalidGrant()
            return next_record

    @asynccontextmanager
    async def _lock(self, key: str) -> c.AsyncIterator[str]:
        """Renew a token-owned grant/comment lock without deleting a successor.

        Args:
            key:
                Grant or comment lock key.

        Yields:
            Opaque owner token used by Lua mutation fences.

        Raises:
            HTTPException:
                If another operation owns this lock.
        """
        token = secrets.token_urlsafe(nbytes=18)
        if not await self.redis.set(key=key, value=token, ex=30, nx=True):
            raise HTTPException(409, "GitHub authorization is busy; retry.")

        async def renew() -> None:
            """Keep the owned lock alive; mutation scripts remain authoritative."""
            while True:
                await asyncio.sleep(10)
                with suppress(Exception):
                    await self.redis.execute(
                        command=["EVAL", _RENEW, "1", key, token, "30"]
                    )

        task = asyncio.create_task(renew())
        try:
            yield token
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            with suppress(Exception):
                await self.redis.execute(command=["EVAL", _DELETE, "1", key, token])

    def _seal(self, record: _Grant, config: _Config) -> str:
        """Encrypt tokens with credential/generation-bound authenticated data.

        Args:
            record:
                Verified token record.
            config:
                Encryption configuration.

        Returns:
            Compatible encrypted JSON envelope.
        """
        iv = secrets.token_bytes(nbytes=12)
        plaintext = json.dumps(
            record, separators=(",", ":"), ensure_ascii=False
        ).encode()
        encrypted = AESGCM(config.key).encrypt(
            iv, plaintext, f"{record['credentialHash']}:{record['generation']}".encode()
        )
        return json.dumps(
            {
                "version": 1,
                "generation": record["generation"],
                "iv": _encode(value=iv),
                "ciphertext": _encode(value=encrypted),
            },
            separators=(",", ":"),
        )

    def _unseal(self, raw: str, config: _Config, credential_hash: str) -> _Grant:
        """Authenticate encrypted authority before making it available to any caller.

        Args:
            raw:
                Persisted envelope.
            config:
                Current App identity and key.
            credential_hash:
                Required credential binding.

        Returns:
            Authenticated grant record.

        Raises:
            InvalidGrant:
                If integrity, identity, or configuration binding fails.
        """
        try:
            envelope = json.loads(raw)
            if (
                type(envelope["version"]) is not int
                or envelope["version"] != 1
                or not envelope["generation"]
            ):
                raise InvalidGrant()
            plain = AESGCM(config.key).decrypt(
                _decode(value=envelope["iv"]),
                _decode(value=envelope["ciphertext"]),
                f"{credential_hash}:{envelope['generation']}".encode(),
            )
            record = json.loads(plain)
            if (
                type(record["version"]) is not int
                or record["version"] != 1
                or record["credentialHash"] != credential_hash
                or record["generation"] != envelope["generation"]
                or record["clientId"] != config.client_id
            ):
                raise InvalidGrant()
            return t.cast(_Grant, record)
        except Exception as error:
            raise InvalidGrant() from error

    async def _verify_user(
        self, config: _Config, access: str, expected: _Grant | None = None
    ) -> tuple[int, str]:
        """Verify the token belongs to this App and the same immutable GitHub user.

        Args:
            config:
                Exact App identity.
            access:
                User access token.
            expected (optional):
                Previously verified identity for rotation.

        Returns:
            Verified user ID and current login.

        Raises:
            InvalidGrant:
                If the provider does not confirm the same token and user.
        """
        response = await self._request(
            url="https://api.github.com/user", authorization=f"Bearer {access}"
        )
        if not response.is_success:
            raise InvalidGrant()
        user = response.json()
        checked = await self._request(
            url=f"https://api.github.com/applications/{config.client_id}/token",
            method="POST",
            authorization=self._basic(config=config),
            body={"access_token": access},
        )
        if not checked.is_success:
            raise InvalidGrant()
        data = checked.json()
        app = data.get("app") or {}
        matched_user = data.get("user") or {}
        if (
            not user.get("id")
            or not user.get("login")
            or data.get("token") != access
            or app.get("client_id") != config.client_id
            or ("id" in app and str(app["id"]) != config.app_id)
            or matched_user.get("id") != user["id"]
            or matched_user.get("login", "").lower() != user["login"].lower()
            or (
                expected
                and (
                    user["id"] != expected["userId"]
                    or user["login"].lower() != expected["contributor"].lower()
                )
            )
        ):
            raise InvalidGrant()
        return user["id"], user["login"]

    async def retain_user_grant(
        self,
        access_token: str,
        refresh_token: str,
        access_expires: int,
        refresh_expires: int,
        credential: str,
        expected_login: str,
    ) -> None:
        """Verify and retain renewable authority without persisting plaintext tokens.

        Args:
            access_token:
                App user access token.
            refresh_token:
                Renewable refresh token.
            access_expires:
                Access expiry in Unix milliseconds.
            refresh_expires:
                Refresh expiry in Unix milliseconds.
            credential:
                Opaque worker credential.
            expected_login:
                Independently verified onboarding login.

        Raises:
            InvalidGrant:
                If login or renewable expiry does not match.
        """
        config = self._config()
        await self.verify_app_installation()
        user_id, login = await self._verify_user(config=config, access=access_token)
        now = int(time.time() * 1000)
        if (
            login.lower() != expected_login.lower()
            or not refresh_token
            or access_expires <= now
            or refresh_expires <= now
        ):
            raise InvalidGrant()
        record: _Grant = {
            "version": 1,
            "generation": secrets.token_urlsafe(nbytes=18),
            "userId": user_id,
            "contributor": login,
            "credentialHash": hashlib.sha256(credential.encode()).hexdigest(),
            "clientId": config.client_id,
            "access": access_token,
            "refresh": refresh_token,
            "accessExpires": access_expires,
            "refreshExpires": refresh_expires,
        }
        await self._persist(record=record, config=config)

    async def _persist(
        self, record: _Grant, config: _Config, expected_generation: str = ""
    ) -> None:
        """Commit an envelope only while its credential, generation, and lock agree.

        Args:
            record:
                Verified grant record.
            config:
                Encryption configuration.
            expected_generation (optional):
                Required old generation for replacement.

        Raises:
            InvalidGrant:
                If authority was revoked or its generation changed.
            HTTPException:
                If lock ownership changed during persistence.
        """
        digest = record["credentialHash"]
        key = f"{_PREFIX}{digest}"
        async with self._lock(key=f"{key}:lock") as token:
            current = await self.redis.get(key)
            generation = (
                json.loads(current)["generation"] if isinstance(current, str) else ""
            )
            if expected_generation and generation != expected_generation:
                raise InvalidGrant()
            ttl = max(
                1,
                min(
                    2592000,
                    math.ceil((record["refreshExpires"] - time.time() * 1000) / 1000),
                ),
            )
            outcome = await self.redis.execute(
                command=[
                    "EVAL",
                    _PERSIST,
                    "3",
                    f"{key}:lock",
                    f"{_CREDENTIAL}{digest}",
                    key,
                    token,
                    generation,
                    self._seal(record=record, config=config),
                    str(ttl),
                ]
            )
            if outcome == -1:
                raise InvalidGrant()
            if outcome != 1:
                raise HTTPException(409, "GitHub authorization changed; retry.")

    async def verify_app_installation(self) -> None:
        """Verify exact selected-repository permissions and revoke temporary access.

        Raises:
            HTTPException:
                If installation identity, scope, or token cleanup fails.
        """
        config = self._config()
        authorization = f"Bearer {self._jwt(config=config)}"
        response = await self._request(
            url=f"https://api.github.com/app/installations/{config.installation_id}",
            authorization=authorization,
        )
        if not response.is_success:
            raise HTTPException(503, "GitHub App installation could not be verified.")
        installation = response.json()
        if (
            str(installation.get("id")) != config.installation_id
            or installation.get("account", {}).get("login", "").lower() != "euroeval"
            or installation.get("repository_selection") != "selected"
            or installation.get("permissions") != _PERMISSIONS
        ):
            raise HTTPException(
                503, "GitHub App installation permissions or owner are invalid."
            )
        minted = await self._request(
            url=f"https://api.github.com/app/installations/{config.installation_id}/access_tokens",
            method="POST",
            authorization=authorization,
            body={"permissions": _PERMISSIONS},
        )
        data = minted.json()
        token = data.get("token")
        if not minted.is_success or not isinstance(token, str) or not token:
            raise HTTPException(
                503, "GitHub App repository access could not be verified."
            )
        try:
            if data.get("permissions") != _PERMISSIONS:
                raise HTTPException(
                    503, "GitHub App installation permissions are invalid."
                )
            repositories = []
            url = (
                data.get("repositories_url")
                or "https://api.github.com/installation/repositories"
            )
            for page in range(1, 101):
                listed = await self._request(
                    url=f"{url}{'&' if '?' in url else '?'}per_page=100&page={page}",
                    authorization=f"Bearer {token}",
                )
                if not listed.is_success:
                    raise HTTPException(
                        503, "GitHub App repository access could not be verified."
                    )
                batch = listed.json().get("repositories", [])
                repositories.extend(batch)
                if len(batch) < 100:
                    break
            else:
                raise HTTPException(
                    503, "GitHub App repository history is too large to verify."
                )
            if (
                len(repositories) != 1
                or str(repositories[0].get("id")) != config.repository_id
                or repositories[0].get("full_name", "").lower() != "euroeval/euroeval"
            ):
                raise HTTPException(
                    503, "GitHub App installation must select only EuroEval/EuroEval."
                )
        finally:
            try:
                revoked = await self._request(
                    url="https://api.github.com/installation/token",
                    method="DELETE",
                    authorization=f"Bearer {token}",
                )
            except httpx.HTTPError as error:
                raise HTTPException(
                    503, "Temporary GitHub App token could not be revoked."
                ) from error
            if not revoked.is_success and revoked.status_code != 404:
                raise HTTPException(
                    503, "Temporary GitHub App token could not be revoked."
                )

    def _jwt(self, config: _Config) -> str:
        """Sign the short-lived GitHub App JWT using its PKCS8 RSA key.

        Args:
            config:
                App issuer and private key.

        Returns:
            Signed RS256 assertion.

        Raises:
            HTTPException:
                If the configured key is not a valid PKCS8 RSA key.
        """
        try:
            if not config.private_key.strip().startswith("-----BEGIN PRIVATE KEY-----"):
                raise HTTPException(
                    503, "GitHub App private key must be a valid PKCS8 key."
                )
            key = serialization.load_pem_private_key(
                config.private_key.encode(), password=None
            )
            if not isinstance(key, rsa.RSAPrivateKey):
                raise HTTPException(503, "GitHub App private key must be RSA.")
            now = int(time.time())
            header = _encode(value=b'{"alg":"RS256","typ":"JWT"}')
            payload = _encode(
                value=json.dumps(
                    {"iat": now - 30, "exp": now + 540, "iss": config.app_id},
                    separators=(",", ":"),
                ).encode()
            )
            content = f"{header}.{payload}"
            signature = key.sign(content.encode(), padding.PKCS1v15(), hashes.SHA256())
            return f"{content}.{_encode(value=signature)}"
        except (ValueError, TypeError) as error:
            raise HTTPException(
                503, "GitHub App private key must be a valid PKCS8 key."
            ) from error

    async def revoke_user_grant(self, identity: WorkerIdentity) -> None:
        """Remove local authority first and retain encrypted retryable remote cleanup.

        Args:
            identity:
                Authenticated identity, or hash of a pending revocation.

        Raises:
            HTTPException:
                If lock fencing or remote revocation remains unconfirmed.
        """
        try:
            config = self._config()
        except HTTPException:
            config = None
        key = f"{_PREFIX}{identity['hash']}"
        pending_key = f"{key}:revocation"
        async with self._lock(key=f"{key}:lock") as token:
            encrypted = await self.redis.get(key)
            if isinstance(encrypted, str):
                if config is None:
                    await self.redis.set(key=pending_key, value=encrypted, ex=2592000)
                else:
                    with suppress(InvalidGrant):
                        grant = self._unseal(
                            raw=encrypted,
                            config=config,
                            credential_hash=identity["hash"],
                        )
                        pending: _Grant = {
                            **grant,
                            "generation": secrets.token_urlsafe(nbytes=18),
                            "userId": 0,
                            "contributor": "",
                            "refresh": "",
                            "accessExpires": int(time.time() * 1000),
                            "refreshExpires": int(time.time() * 1000) + 2592000000,
                        }
                        await self.redis.set(
                            key=pending_key,
                            value=self._seal(record=pending, config=config),
                            ex=2592000,
                        )
            await self.redis.delete(f"{_CREDENTIAL}{identity['hash']}")
            removed = await self.redis.execute(
                command=[
                    "EVAL",
                    _REMOVE,
                    "3",
                    f"{key}:lock",
                    key,
                    f"{_CREDENTIAL}{identity['hash']}",
                    token,
                    "",
                ]
            )
            if removed != 1:
                raise HTTPException(
                    503,
                    "Credential revocation could not confirm ownership; retry revoke.",
                )
        stored = await self.redis.get(pending_key)
        if config is None or not isinstance(stored, str):
            return
        try:
            pending = self._unseal(
                raw=stored, config=config, credential_hash=identity["hash"]
            )
        except InvalidGrant:
            return
        response = await self._request(
            url=f"https://api.github.com/applications/{config.client_id}/token",
            method="DELETE",
            authorization=self._basic(config=config),
            body={"access_token": pending["access"]},
        )
        if response.is_success or response.status_code == 404:
            await self.redis.delete(pending_key)
        else:
            raise HTTPException(
                503, "GitHub App token revocation is pending; retry revoke."
            )
