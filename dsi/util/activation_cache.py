"""RAM-resident LRU cache for SAE activations.

CLAUDE.md §4 / v2 §5: with 1 TB RAM available, hold all four hookpoints
x 100K samples x 5120 features (~170 GB) in RAM simultaneously rather than
disk-paging through SAE caches. Query latency target: < 100us per (exp, sample).

Cache stores per-key tensors keyed by `(exp, sample_id)` or `(exp, sample_id, hookpoint)`.
Backed by a OrderedDict (move-to-end on hit) with byte-budgeted eviction.
Flush-to-disk path is provided for evictions and persistent storage.

Public API:
    cache = ActivationCache(max_bytes=200 * 1024**3, flush_dir=...)
    cache.put((exp, sample_id, hp), tensor)
    t = cache.get((exp, sample_id, hp))      # None on miss
    cache.preload_dir(dir, glob='*.sae.pt')   # bulk load
    cache.flush()                              # write dirty entries to disk
"""

from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Hashable, Iterator, Optional

import torch


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _obj_bytes(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return _tensor_bytes(x)
    if isinstance(x, dict):
        return sum(_obj_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_obj_bytes(v) for v in x)
    if isinstance(x, (bytes, bytearray)):
        return len(x)
    return 64  # cheap default for scalars / ints / strings


class ActivationCache:
    """Bytes-budgeted in-memory LRU. Thread-safe under a single lock.

    `max_bytes` clamps the resident set; on insert overflow the oldest entries
    are evicted (and written to `flush_dir` if provided and dirty).
    """

    DEFAULT_MAX_BYTES = 200 * 1024**3  # 200 GB hard ceiling per CLAUDE.md §4

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        flush_dir: Optional[str | os.PathLike] = None,
    ) -> None:
        self._d: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._sizes: dict[Hashable, int] = {}
        self._dirty: set[Hashable] = set()
        self._used = 0
        self._max = int(max_bytes)
        self._lock = threading.Lock()
        self.flush_dir = Path(flush_dir) if flush_dir is not None else None
        if self.flush_dir is not None:
            self.flush_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def used_bytes(self) -> int:
        return self._used

    @property
    def n_entries(self) -> int:
        return len(self._d)

    def stats(self) -> dict:
        return {
            "n_entries": self.n_entries,
            "used_bytes": self._used,
            "used_gb": self._used / 1024**3,
            "max_gb": self._max / 1024**3,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._d

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self._hits += 1
                return self._d[key]
            self._misses += 1
            return default

    def put(self, key: Hashable, value: Any, *, dirty: bool = False) -> None:
        size = _obj_bytes(value)
        with self._lock:
            if key in self._d:
                self._used -= self._sizes[key]
                del self._d[key]
            self._d[key] = value
            self._sizes[key] = size
            self._used += size
            if dirty:
                self._dirty.add(key)
            self._evict_until_under_budget_locked()

    def _evict_until_under_budget_locked(self) -> None:
        while self._used > self._max and self._d:
            k, v = self._d.popitem(last=False)
            sz = self._sizes.pop(k, 0)
            self._used -= sz
            if k in self._dirty and self.flush_dir is not None:
                try:
                    self._flush_one_locked(k, v)
                except Exception:
                    pass
                self._dirty.discard(k)
            self._evictions += 1

    def _key_to_path(self, key: Hashable) -> Path:
        assert self.flush_dir is not None
        if isinstance(key, tuple):
            stem = "__".join(str(x) for x in key)
        else:
            stem = str(key)
        safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in stem)
        return self.flush_dir / f"{safe}.pt"

    def _flush_one_locked(self, key: Hashable, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            torch.save(value, self._key_to_path(key))
        else:
            self._key_to_path(key).with_suffix(".pkl").write_bytes(pickle.dumps(value))

    def flush(self) -> int:
        """Write all dirty entries to flush_dir; clear the dirty set."""
        if self.flush_dir is None:
            return 0
        n = 0
        with self._lock:
            for k in list(self._dirty):
                self._flush_one_locked(k, self._d[k])
                n += 1
            self._dirty.clear()
        return n

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._sizes.clear()
            self._dirty.clear()
            self._used = 0

    def keys(self) -> Iterator[Hashable]:
        with self._lock:
            return iter(list(self._d.keys()))

    def preload_dir(self, src_dir: str | os.PathLike, glob: str = "*.sae.pt",
                    key_fn=None) -> int:
        """Bulk-load SAE activation tensors. `key_fn(path) -> hashable` keys; default uses path stem."""
        src = Path(src_dir)
        loaded = 0
        for p in sorted(src.rglob(glob)):
            try:
                t = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            key = key_fn(p) if key_fn else (str(p.relative_to(src)),)
            self.put(key, t)
            loaded += 1
        return loaded


_GLOBAL: Optional[ActivationCache] = None


def get_global_cache(max_bytes: Optional[int] = None) -> ActivationCache:
    """Lazy-init a process-wide cache. Override `max_bytes` on first call only."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ActivationCache(
            max_bytes=max_bytes if max_bytes is not None else ActivationCache.DEFAULT_MAX_BYTES
        )
    return _GLOBAL
