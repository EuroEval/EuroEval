"""Atomic Redis state operations used by the volunteer worker."""
from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any

from .core import BrokerError, Runtime, random_token, sha256

LEASE_TTL_MAX = 21600
LEASE_TOMBSTONE_TTL = 2592000
ISSUE_MUTEX_TTL = 60
ISSUE_MUTEX_RENEW_INTERVAL = 10
MAX_RESULT_RESERVATIONS = 1024

def lease_ttl(runtime: Runtime) -> int:
    try: value = float(runtime.value("VOLUNTEER_LEASE_SECONDS", "1800"))
    except ValueError: return 1800
    return max(60, min(LEASE_TTL_MAX, math.floor(value))) if math.isfinite(value) else 1800

def issue_lease_key(issue: int, language: str) -> str: return f"euroeval:worker:lease:{issue}:{language}"
def result_reservations_key(lease_id: str) -> str: return f"euroeval:worker:reservations:{lease_id}"
def lease_key(lease_id: str) -> str: return f"euroeval:worker:lease-id:{lease_id}"
def mutex_key(issue: int) -> str: return f"euroeval:worker:mutex:{issue}"
def identity_key(identity: str) -> str: return f"euroeval:worker:record-identity:{sha256(identity)}"

async def get_lease_for_issue(runtime: Runtime, issue: int, language: str) -> object | None:
    return await runtime.get(issue_lease_key(issue, language))
async def get_lease_by_id(runtime: Runtime, lease_id: str) -> object | None:
    return await runtime.get(lease_key(lease_id))

async def reclaim_expired_lease(runtime: Runtime, lease: dict[str, Any]) -> bool:
    raw = await runtime.redis("SMEMBERS", result_reservations_key(lease["lease_id"]))
    if not isinstance(raw, list) or len(raw) > MAX_RESULT_RESERVATIONS: return False
    records = []
    for value in raw:
        try:
            item = json.loads(value)
            if not item.get("identity") or not __import__('re').fullmatch(r"[0-9a-f]{64}", item.get("digest", "")): return False
            records.append({"identity":item["identity"],"digest":item["digest"]})
        except (ValueError, AttributeError, TypeError): return False
    expiry = __import__('datetime').datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00")).timestamp()
    if expiry > __import__('time').time(): return False
    keys=[issue_lease_key(lease["issue_number"],lease["language"]),lease_key(lease["lease_id"]),result_reservations_key(lease["lease_id"]),*[identity_key(r["identity"]) for r in records]]
    script="local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.lease_id ~= ARGV[1] or item.expires_at ~= ARGV[2] or tonumber(ARGV[3]) > tonumber(redis.call('TIME')[1]) then return 0 end; local byid=redis.call('GET',KEYS[2]); if not byid or cjson.decode(byid).lease_id ~= ARGV[1] then return 0 end; local records=cjson.decode(ARGV[4]); for i=4,#KEYS do local value=redis.call('GET',KEYS[i]); if value then local owned=cjson.decode(value); local record=records[i-3]; if owned.lease_id == ARGV[1] and owned.digest == record.digest and owned.identity == record.identity then redis.call('DEL',KEYS[i]) end end end; redis.call('DEL',KEYS[3]); redis.call('DEL',KEYS[1]); redis.call('DEL',KEYS[2]); return 1"
    out=await runtime.redis("EVAL",script,str(len(keys)),*keys,lease["lease_id"],lease["expires_at"],str(int(expiry)),json.dumps(records,separators=(",",":")))
    return str(out)=="1"

async def acquire_issue_mutex(runtime: Runtime, issue: int) -> str | None:
    token=random_token(12)
    return token if await runtime.set(mutex_key(issue),token,ISSUE_MUTEX_TTL,True) else None

async def release_issue_mutex(runtime: Runtime, issue: int, token: str, timeout: float | None = None) -> None:
    await runtime.redis("EVAL","if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('DEL',KEYS[1]) else return 0 end","1",mutex_key(issue),token)
async def renew_issue_mutex(runtime: Runtime, issue: int, token: str, timeout: float | None = None) -> bool:
    return str(await runtime.redis("EVAL","if redis.call('GET',KEYS[1]) == ARGV[1] then return redis.call('EXPIRE',KEYS[1],ARGV[2]) else return 0 end","1",mutex_key(issue),token,str(ISSUE_MUTEX_TTL)))=="1"

@dataclass
class IssueMutex:
    token: str
    runtime: Runtime
    issue: int
    _lost: bool = False
    _stopped: bool = False
    _task: asyncio.Task[None] | None = None
    async def _renew_loop(self) -> None:
        while not self._stopped:
            await asyncio.sleep(ISSUE_MUTEX_RENEW_INTERVAL)
            if not self._stopped and not await renew_issue_mutex(self.runtime,self.issue,self.token): self._lost=True
    async def assert_owned(self) -> None:
        if self._lost or not await renew_issue_mutex(self.runtime,self.issue,self.token):
            self._lost=True; raise BrokerError(409,"Issue mutex was lost before a mutation.")
    async def release(self, timeout: float | None = None) -> None:
        self._stopped=True
        if self._task: self._task.cancel()
        if not self._lost and await renew_issue_mutex(self.runtime,self.issue,self.token): await release_issue_mutex(self.runtime,self.issue,self.token)

async def acquire_renewable_issue_mutex(runtime: Runtime, issue: int) -> IssueMutex | None:
    token=await acquire_issue_mutex(runtime,issue)
    if token is None:return None
    lock=IssueMutex(token,runtime,issue); lock._task=asyncio.create_task(lock._renew_loop()); return lock

async def reserve_result_identity(runtime: Runtime,key: str,reservation: str,digest: str,lease_id: str,ttl: int,reservations_key: str,reservation_entry: str) -> str:
    script="local function track() if redis.call('SISMEMBER',KEYS[2],ARGV[5]) == 0 and redis.call('SCARD',KEYS[2]) >= tonumber(ARGV[6]) then return 0 end; redis.call('SADD',KEYS[2],ARGV[5]); redis.call('EXPIRE',KEYS[2],ARGV[7]); return 1 end; local current=redis.call('GET',KEYS[1]); if not current then if track()==0 then return 'full' end; redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[4]); return 'reserved' end; local item=cjson.decode(current); if item.digest ~= ARGV[2] then return 'busy' end; if item.status == 'uploaded' then if track()==0 then return 'full' end; return 'duplicate' end; if item.lease_id == ARGV[3] then if track()==0 then return 'full' end; return 'retry' end; return 'busy'"
    return str(await runtime.redis("EVAL",script,"2",key,reservations_key,reservation,digest,lease_id,str(ttl),reservation_entry,str(MAX_RESULT_RESERVATIONS),str(LEASE_TOMBSTONE_TTL)))

async def complete_result_identity(runtime: Runtime,key: str,reservation: str,digest: str,lease_id: str,ttl: int) -> bool:
    script="local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.digest ~= ARGV[1] or item.lease_id ~= ARGV[2] then return 0 end; redis.call('SET',KEYS[1],ARGV[3],'EX',ARGV[4]); return 1"
    return str(await runtime.redis("EVAL",script,"1",key,digest,lease_id,reservation,str(ttl)))=="1"
async def abort_result_identity(runtime: Runtime,key: str,digest: str,lease_id: str) -> None:
    await runtime.redis("EVAL","local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.digest == ARGV[1] and item.lease_id == ARGV[2] and item.status == 'uploading' then return redis.call('DEL',KEYS[1]) end; return 0","1",key,digest,lease_id)
async def release_result_reservation(runtime: Runtime,key: str,digest: str,lease_id: str) -> bool:
    script="local current=redis.call('GET',KEYS[1]); if not current then return 1 end; local item=cjson.decode(current); if item.digest ~= ARGV[1] or item.lease_id ~= ARGV[2] then return 0 end; return redis.call('DEL',KEYS[1])"
    return str(await runtime.redis("EVAL",script,"1",key,digest,lease_id))=="1"
async def release_result_reservations(runtime: Runtime,records: list[dict[str,str]],lease_id: str) -> bool:
    keys=[identity_key(r["identity"]) for r in records]+[result_reservations_key(lease_id)]
    script="local records=cjson.decode(ARGV[1]); for i=1,#KEYS-1 do local current=redis.call('GET',KEYS[i]); if current then local item=cjson.decode(current); local record=records[i]; if item.digest ~= record.digest or item.lease_id ~= ARGV[2] then return 0 end end end; for i=1,#KEYS-1 do redis.call('DEL',KEYS[i]) end; redis.call('DEL',KEYS[#KEYS]); return 1"
    return str(await runtime.redis("EVAL",script,str(len(keys)),*keys,json.dumps(records,separators=(",",":")),lease_id))=="1"
async def put_lease(runtime: Runtime,lease: dict[str,Any],timeout: float | None = None) -> bool:
    ttl = max(60, math.ceil(__import__('datetime').datetime.fromisoformat(lease['expires_at'].replace('Z', '+00:00')).timestamp() - __import__('time').time())) + LEASE_TOMBSTONE_TTL
    script="local issue=redis.call('GET',KEYS[1]); local byid=redis.call('GET',KEYS[2]); if issue or byid then if not issue or not byid or issue ~= ARGV[1] or byid ~= ARGV[1] then return 0 end; local item=cjson.decode(issue); if item.lease_id ~= ARGV[3] then return 0 end; return 1 end; redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2]); redis.call('SET',KEYS[2],ARGV[1],'EX',ARGV[2]); return 1"
    raw=json.dumps(lease,separators=(",",":"),ensure_ascii=False)
    return str(await runtime.redis("EVAL",script,"2",issue_lease_key(lease['issue_number'],lease['language']),lease_key(lease['lease_id']),raw,str(ttl),lease['lease_id']))=="1"
async def save_lease(runtime: Runtime,lease: dict[str,Any]) -> bool:
    ttl=max(60,math.ceil(__import__('datetime').datetime.fromisoformat(lease['expires_at'].replace('Z','+00:00')).timestamp()-__import__('time').time()))+LEASE_TOMBSTONE_TTL
    script="local current=redis.call('GET',KEYS[1]); if not current then return 0 end; local item=cjson.decode(current); if item.lease_id ~= ARGV[1] then return 0 end; redis.call('SET',KEYS[1],ARGV[2],'EX',ARGV[3]); redis.call('SET',KEYS[2],ARGV[2],'EX',ARGV[3]); return 1"
    raw=json.dumps(lease,separators=(",",":"),ensure_ascii=False)
    return str(await runtime.redis("EVAL",script,"2",issue_lease_key(lease['issue_number'],lease['language']),lease_key(lease['lease_id']),lease['lease_id'],raw,str(ttl)))=="1"
async def delete_lease(runtime: Runtime,lease: dict[str,Any],timeout: float | None = None) -> None:
    script="local current=redis.call('GET',KEYS[1]); if current then local item=cjson.decode(current); if item.lease_id == ARGV[1] then redis.call('DEL',KEYS[1]) end end; local byid=redis.call('GET',KEYS[2]); if byid then local item=cjson.decode(byid); if item.lease_id == ARGV[1] then item.released=true; redis.call('SET',KEYS[2],cjson.encode(item),'EX',ARGV[2]) end end; return 1"
    await runtime.redis("EVAL",script,"2",issue_lease_key(lease['issue_number'],lease['language']),lease_key(lease['lease_id']),lease['lease_id'],str(LEASE_TOMBSTONE_TTL))
