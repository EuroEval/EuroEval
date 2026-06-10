# EuroEval Results Storage Architecture

## Overview

EuroEval now uses **hf-mount** for live synchronization between the HF bucket and
local storage, with automatic backups to pCloud.

## Storage Hierarchy

```
HF Bucket (source of truth)
hf://buckets/EuroEval/raw-results/
├── meta_llama_Llama-3-8B.jsonl
├── openai_gpt-4.jsonl
└── ... (77k+ records in ~1000 per-model files)
    │
    │ hf-mount (NFS backend, read-write)
    ↓
Mount Point (configurable)
~/.local/share/euroeval-results/
├── meta_llama_Llama-3-8B.jsonl
├── openai_gpt-4.jsonl
└── ... (live mirror of bucket)
    │
    │ Copy to cache for processing
    ↓
Processing Cache
.euroeval_cache/results/
├── meta_llama_Llama-3-8B.jsonl
└── ... (temporary, can be deleted)
    │
    │ Periodic backup
    ↓
pCloud Backup
~/pCloud Drive/data/euroeval_backup/
├── results_20260610_150000.tar.gz
└── ... (rotated backups, <1 GB total)
```

## Configuration

### Mount Point

Default: `~/.local/share/euroeval-results`

Override with env var:

```bash
export EUROEVAL_MOUNT_POINT=/path/to/custom/location
```

**Requirements:**

- Must be `.gitignore`'d (already in repo `.gitignore` as `euroeval-results/`)
- Should be on persistent storage (not tmpfs)
- Can be on pCloud or external drive for large datasets

### HF Token

Required for bucket access:

```bash
export HF_TOKEN=your_huggingface_token
# Or add to .env file
```

## Installation

```bash
# macOS
brew install hf-mount

# Linux
# Download from https://github.com/huggingface/hf-mount/releases
```

If `hf-mount` is not available, system falls back to `huggingface_hub` sync (slower).

## Usage

### Normal Flow (Harvest → Deploy)

```bash
uv run src/scripts/collect_evaluation_results.py
```

**What happens:**

1. Harvest results from GitHub issues
2. Upload to HF bucket (per-model files)
3. Mount bucket with `hf-mount`
4. Copy to `.euroeval_cache/results/`
5. Rebuild `results.tar.gz` (local archive)
6. Merge any new results
7. Generate leaderboards
8. ✅ Verify leaderboards (sanity check)
9. Deploy to Vercel
10. **Create backup to pCloud**
11. Close GitHub issues

### Manual Mount

```bash
# Mount bucket
hf-mount-nfs bucket buckets/EuroEval/raw-results ~/.local/share/euroeval-results \
  --hf-token $HF_TOKEN

# Browse results
ls -lh ~/.local/share/euroeval-results/

# Unmount
umount ~/.local/share/euroeval-results
```

## Backup Rotation

Backups are automatically created after each successful harvest:

- Location: `~/pCloud Drive/data/euroeval_backup/`
- Format: `results_YYYYMMDD_HHMMSS.tar.gz`
- Size limit: 1 GB total (oldest removed first)
- Contains: All per-model JSONL files + empty processed file

## Recovery

### New Machine Setup

1. Install `hf-mount`
2. Set `HF_TOKEN` in `.env`
3. Run `generate_leaderboards.py` — bucket will be mounted automatically

**OR** (if hf-mount unavailable):

1. Copy latest backup from pCloud
2. Place as `results.tar.gz` in repo root
3. Run `generate_leaderboards.py`

### Disaster Recovery

If bucket is corrupted/deleted:

```bash
# Restore from pCloud backup
cp ~/pCloud\ Drive/data/euroeval_backup/results_*.tar.gz results.tar.gz

# Rebuild bucket (manual intervention required)
# Contact maintainer to re-upload from backup
```

## File Locations

| Path | Purpose | Git-tracked? |
|------|---------|--------------|
| `results.tar.gz` | Local archive | ❌ |
| `new_results.jsonl` | Just-harvested (pre-upload) | ❌ |
| `.euroeval_cache/` | Processing cache | ❌ |
| `euroeval-results/` | hf-mount point | ❌ |
| `~/pCloud Drive/data/euroeval_backup/` | Backups | ❌ |
| `src/frontend/csv/` | Generated leaderboards | ❌ |

## Migration from Old System

The old `huggingface_hub` sync logic is still available as fallback. To migrate:

1. Install `hf-mount`
2. Next run will automatically use it
3. Verify backup is created
4. (**Optional**) Delete old sync cache `~/.euroeval_cache/results/`

Rollback: Uninstall `hf-mount` — system falls back automatically.

## Troubleshooting

### Mount fails

```
⚠  hf-mount failed: ...
```

**Fix:** Check `HF_TOKEN` is set and valid. Ensure bucket exists.

### "Not enough results in bucket" warning

**Fix:** This indicates bucket is stale. Check if previous `upload_to_hf()` failed.
Re-run harvest script.

### Mount point not writable

**Fix:** Check disk space. Try different mount point with `EUROEVAL_MOUNT_POINT`.

### Slow file access

**Fix:** First reads require network fetch. Subsequent reads are cached. Clear cache:

```bash
rm -rf .euroeval_cache/
```

## Why hf-mount?

| Scenario | Old (huggingface_hub) | New (hf-mount) |
|----------|----------------------|----------------|
| **New machine** | Download 77k records via API | Mount instantly |
| **Stale bucket** | Detect & warn | Never stale (10s polling) |
| **Sync complexity** | Merge logic, staleness checks | None (live mount) |
| **Disk usage** | 53 MB tar.gz + cache | Cache only (lazy) |
| **Backup** | Manual from tar.gz | Automatic after harvest |
| **Failure mode** | Silent data loss | Mount fails (obvious) |

## Technical Details

- **Backend:** NFS (no root required)
- **Consistency:** 10s polling interval (eventual consistency)
- **Caching:** Local chunk cache in `~/.hf-mount-cache/`
- **Fallback:** `huggingface_hub.HfApi().sync_bucket()` if hf-mount unavailable

## See Also

- [hf-mount docs](https://github.com/huggingface/hf-mount)
- `src/leaderboards/hf_mount.py` — Mount management
- `src/leaderboards/result_loading.py` — Sync logic
- `src/scripts/collect_evaluation_results.py` — Harvest flow
