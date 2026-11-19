#!/usr/bin/env python3
"""Resource reservations for concurrent training (standard library only).

Admissions use available memory minus the part of each active reservation that
has not yet appeared in telemetry. Thus two just-launched processes cannot both
claim the same free VRAM/RAM. No batch size or scientific setting is changed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

GIB = 1024 ** 3
MIB = 1024 ** 2
_PROC_PATH_CACHE = {}


def proc_directory(pid, proc_root="/proc"):
    """Map a local PID to /proc even if it is mounted in an ancestor namespace.

    /proc/6 may describe the host's PID 6 while os.kill(6, ...) addresses a
    different child PID namespace. Match both namespace inode and NSpid before
    accepting a path; a host process must never prove a local child is alive.
    """
    pid, root = int(pid), Path(proc_root)
    try:
        if int((root / "self").readlink().name) == os.getpid():
            path = root / str(pid)
            return path if path.exists() else None
        namespace = (root / "self/ns/pid").readlink()
        key = (str(root), pid)
        cached = _PROC_PATH_CACHE.get(key)
        candidates = [cached] if cached else []
        candidates.extend(path for path in root.iterdir() if path.name.isdigit() and path != cached)
        for path in candidates:
            try:
                if (path / "ns/pid").readlink() != namespace:
                    continue
                values = next(line.split()[1:] for line in (path / "status").read_text().splitlines()
                              if line.startswith("NSpid:"))
                if int(values[-1]) == pid:
                    _PROC_PATH_CACHE[key] = path
                    return path
            except (OSError, StopIteration, ValueError):
                pass
    except (OSError, ValueError):
        pass
    return None


def process_children(pid):
    """Return child PIDs in our namespace, rather than the proc mount's PIDs."""
    directory = proc_directory(pid)
    if directory is None:
        return []
    try:
        outer_children = (directory / f"task/{directory.name}/children").read_text().split()
    except OSError:
        return []
    children = []
    for outer in outer_children:
        try:
            child = directory.parent / outer
            if (child / "ns/pid").readlink() != (directory / "ns/pid").readlink():
                continue
            values = next(line.split()[1:] for line in (child / "status").read_text().splitlines()
                          if line.startswith("NSpid:"))
            local_pid = int(values[-1])
            _PROC_PATH_CACHE[(str(directory.parent), local_pid)] = child
            children.append(local_pid)
        except (OSError, StopIteration, ValueError):
            pass
    return children


def process_rss_bytes(pid):
    """Current Linux process-tree RSS; shared mappings make this conservative."""
    seen, pending, total = set(), [int(pid)], 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            directory = proc_directory(current)
            if directory is None:
                continue
            status = (directory / "status").read_text()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
            pending.extend(process_children(current))
        except (OSError, ValueError):
            pass
    return total


def process_anonymous_rss_bytes(pid):
    """Only non-file RSS can discharge a process's pending private RAM claim.

    The feature memmaps are shared file-backed pages. Their full RSS appearing
    in several children must not discharge several *private* reservations;
    that cache is already budgeted once by the controller.
    """
    seen, pending, total = set(), [int(pid)], 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            directory = proc_directory(current)
            if directory is None:
                continue
            for line in (directory / "status").read_text().splitlines():
                if line.startswith("RssAnon:"):
                    total += int(line.split()[1]) * 1024
            pending.extend(process_children(current))
        except (OSError, ValueError):
            pass
    return total


def memory_snapshot():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.split()[0]) * 1024
    total, available = values["MemTotal"], values["MemAvailable"]
    cgroup = _cgroup_memory_limit()
    if cgroup is not None:
        limit, current = cgroup
        total = min(total, limit)
        available = min(available, max(0, limit - current))
    return {"total_bytes": total, "available_bytes": available,
            "host_total_bytes": values["MemTotal"], "host_available_bytes": values["MemAvailable"],
            "cgroup_memory_limit_bytes": cgroup[0] if cgroup else None,
            "cgroup_memory_current_bytes": cgroup[1] if cgroup else None}


def _cgroup_memory_limit():
    """Honor a container/job memory ceiling when /proc reports the host RAM."""
    candidates = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers == "":
                candidates.append((Path("/sys/fs/cgroup") / relative.lstrip("/"), "memory.max", "memory.current"))
            elif "memory" in controllers.split(","):
                candidates.append((Path("/sys/fs/cgroup/memory") / relative.lstrip("/"), "memory.limit_in_bytes", "memory.usage_in_bytes"))
    except OSError:
        pass
    candidates += [(Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
                   (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")]
    limits = []
    for directory, limit_name, current_name in candidates:
        # Include ancestors: a parent scheduler cgroup may be more restrictive.
        while str(directory).startswith("/sys/fs/cgroup"):
            try:
                raw = (directory / limit_name).read_text().strip()
                if raw != "max":
                    limit, current = int(raw), int((directory / current_name).read_text().strip())
                    if 0 < limit < (1 << 60):
                        limits.append((limit, current))
            except (OSError, ValueError):
                pass
            if directory == Path("/sys/fs/cgroup"):
                break
            directory = directory.parent
    if not limits:
        return None
    limit = min(pair[0] for pair in limits)
    headroom = min(pair[0] - pair[1] for pair in limits)
    return limit, max(0, limit - headroom)


def available_cpu_slots():
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            count = min(count, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError):
        try:
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                count = min(count, max(1, math.floor(quota / period)))
        except (OSError, ValueError):
            pass
    return count


def _nvidia_query(fields, kind="gpu"):
    command = ["nvidia-smi", f"--query-{kind}=" + ",".join(fields), "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    return [[part.strip() for part in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


def nvidia_snapshot():
    """Return physical GPU UUID capacities and process memory, or fail closed."""
    rows = _nvidia_query(("index", "uuid", "memory.total", "memory.free", "memory.used"))
    devices = {}
    for index, uuid, total, free, used in rows:
        devices[uuid] = {"physical_index": int(index), "uuid": uuid,
                         "total_bytes": int(float(total) * MIB),
                         "free_bytes": int(float(free) * MIB),
                         "used_bytes": int(float(used) * MIB), "pid_bytes": {}}
    # Failure of per-PID telemetry is not permission to ignore pending claims.
    # Keep all reservations pending instead, which can serialize large jobs.
    process_error = None
    try:
        for uuid, pid, amount in _nvidia_query(("gpu_uuid", "pid", "used_gpu_memory"), "compute-apps"):
            if uuid in devices and amount not in ("N/A", "[N/A]", "[Not Supported]"):
                devices[uuid]["pid_bytes"][int(pid)] = int(float(amount) * MIB)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        process_error = f"{type(exc).__name__}: {exc}"
    return {"devices": devices, "process_telemetry_error": process_error}


def visible_gpu_mapping(snapshot, requested=None, visible=None):
    """Map CUDA-visible logical indices to physical UUIDs.

    Numeric visibility alone cannot prove CUDA ordering; the runner verifies
    actual logical UUIDs in a separate torch process. UUID visibility is
    unambiguous. MIG is rejected
    explicitly because a parent GPU's free memory is not a slice's capacity.
    """
    devices = snapshot["devices"]
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        tokens = [value.strip() for value in visible.split(",") if value.strip()]
        if any(value.startswith("MIG-") for value in tokens):
            raise RuntimeError("MIG visibility requires slice-level telemetry; use one explicitly provisioned job per slice, not this automatic shared-GPU scheduler.")
        ordered = []
        for token in tokens:
            if token.lstrip("-").isdigit():
                if int(token) < 0:
                    continue
                matches = [gpu for gpu in devices.values() if gpu["physical_index"] == int(token)]
            else:
                matches = [gpu for uuid, gpu in devices.items() if uuid.startswith(token)]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot map CUDA_VISIBLE_DEVICES={token!r} uniquely to nvidia-smi UUIDs.")
            ordered.append(matches[0])
    else:
        ordered = sorted(devices.values(), key=lambda value: value["physical_index"])
    indices = list(range(len(ordered))) if requested is None else list(requested)
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(ordered) for index in indices):
        raise ValueError(f"gpu_ids must be distinct CUDA-visible logical indices in 0..{len(ordered)-1}")
    return {index: ordered[index]["uuid"] for index in indices}


@dataclass
class ResourceRequest:
    host_bytes: int
    gpu_bytes: int = 0
    cpu_slots: int = 1


@dataclass
class Reservation:
    job_id: str
    pid: int
    request: ResourceRequest
    gpu_uuid: str | None = None
    logical_gpu: int | None = None
    started_at: float = 0


def training_request_from_profile(profile, safety_factor=1.35, cpu_threads=4,
                                  workers=0, extra_host_gib=1.0):
    """Stage-specific reservation; full export buffer is NOT charged during train.

    The shared, read-only feature-cache reservation is charged ONCE by the
    ResourceController. This per-run estimate covers measured RSS (whole tree),
    full-epoch val/train metric buffers and a fixed host headroom. Export and
    analysis must run only after all training children have exited.
    """
    if safety_factor < 1 or extra_host_gib < 0:
        raise ValueError("safety_factor >= 1 and extra_host_gib >= 0 required")
    measured = max(int(profile.get('host_peak_rss_bytes') or 0),
                   int(profile.get('host_process_tree_peak_sampled_rss_bytes') or 0))
    if measured <= 0:
        raise ValueError('No measured host RSS in profile')
    metrics = max(int(profile.get('host_epoch_metric_buffer_budget_bytes') or 0),
                  int(profile.get('host_inference_buffer_budget_bytes') or 0))
    prefetch = int(profile.get('host_worker_prefetch_budget_bytes') or 0)
    # The larger export_buffer is intentionally excluded from concurrent phase.
    host = math.ceil((measured + metrics + prefetch + int(extra_host_gib * GIB)) * safety_factor)
    gpu = max(int(profile.get('gpu_admission_estimate_bytes') or 0),
              int(profile.get('gpu_peak_reserved_bytes') or 0) + int(profile.get('gpu_context_baseline_bytes') or 0),
              int(profile.get('gpu_driver_delta_bytes') or 0))
    if str(profile.get('device', 'cpu')).startswith('cuda') and gpu <= 0:
        raise ValueError('No positive CUDA memory profile')
    return ResourceRequest(host_bytes=max(host, 256*MIB), gpu_bytes=math.ceil(gpu*safety_factor),
                           cpu_slots=max(1, int(cpu_threads)) + max(0, int(workers)))


def request_from_profile(profile, safety_factor=1.35, cpu_threads=4, workers=0):
    rss = max(int(profile.get("host_peak_rss_bytes") or 0), int(profile.get("host_admission_estimate_bytes") or 0))
    if rss <= 0:
        raise ValueError("Resource profile has no positive host_peak_rss_bytes")
    gpu_peak = max(int(profile.get("gpu_peak_reserved_bytes") or 0) + int(profile.get("gpu_context_baseline_bytes") or 0),
                   int(profile.get("gpu_driver_delta_bytes") or 0), int(profile.get("gpu_admission_estimate_bytes") or 0))
    if str(profile.get("device", "cpu")).startswith("cuda") and gpu_peak <= 0:
        raise ValueError("CUDA resource profile has no positive measured GPU memory")
    return ResourceRequest(host_bytes=math.ceil(max(rss, 256 * MIB) * safety_factor),
                           gpu_bytes=math.ceil(gpu_peak * safety_factor),
                           cpu_slots=max(1, int(cpu_threads)) + max(0, int(workers)))


class ResourceController:
    def __init__(self, max_parallel=4, cpu_budget=None, cpu_parallel_limit=2,
                 gpu_reserve_gib=2, ram_reserve_gib=2, reserve_fraction=0.10,
                 gpu_mapping=None, event_path=None, shared_cache_budget_bytes=0):
        available_cpus = available_cpu_slots()
        self.max_parallel = max(1, int(max_parallel))
        self.cpu_budget = int(cpu_budget or max(1, available_cpus - int(available_cpus >= 8)))
        self.cpu_parallel_limit = max(1, int(cpu_parallel_limit))
        self.gpu_reserve_bytes, self.ram_reserve_bytes = int(gpu_reserve_gib * GIB), int(ram_reserve_gib * GIB)
        self.reserve_fraction = reserve_fraction
        self.gpu_mapping = dict(gpu_mapping or {})
        self.active = {}
        self.event_path = Path(event_path) if event_path else None
        # Charged a single time across all concurrent jobs, including pending
        # launches. Conservative (it may overlap OS page cache) but not per-run.
        self.shared_cache_budget_bytes = max(0, int(shared_cache_budget_bytes))

    def record(self, event, **values):
        payload = {"timestamp": time.time(), "event": event, **values}
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        return payload

    def snapshot(self):
        snapshot = {"host": memory_snapshot(), "process_rss": {r.pid: process_rss_bytes(r.pid) for r in self.active.values()},
                    "process_private_rss": {r.pid: process_anonymous_rss_bytes(r.pid) for r in self.active.values()},
                    "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
        if self.gpu_mapping:
            snapshot["gpu"] = nvidia_snapshot()
        return snapshot

    def admission(self, request, snapshot, logical_gpu=None):
        """Return (allowed, reason, details), without changing reservations."""
        if len(self.active) >= self.max_parallel:
            return False, "max_parallel", {}
        if logical_gpu is None and len(self.active) >= self.cpu_parallel_limit:
            return False, "cpu_parallel_limit", {}
        active_cpu = sum(item.request.cpu_slots for item in self.active.values())
        if active_cpu + request.cpu_slots > self.cpu_budget:
            return False, "cpu_budget", {"active_cpu_slots": active_cpu, "requested": request.cpu_slots, "budget": self.cpu_budget}
        host = snapshot["host"]
        reserve = max(self.ram_reserve_bytes, int(host["total_bytes"] * self.reserve_fraction))
        observed = snapshot.get("process_rss", {})
        private = snapshot.get("process_private_rss", {})
        def private_observed(item):
            if item.pid in private or str(item.pid) in private:
                return private.get(item.pid, private.get(str(item.pid), 0))
            rss = observed.get(item.pid, observed.get(str(item.pid), 0))
            return max(0, rss - self.shared_cache_budget_bytes)
        pending = sum(max(0, item.request.host_bytes - private_observed(item))
                      for item in self.active.values())
        usable_host = host["available_bytes"] - reserve - self.shared_cache_budget_bytes - pending
        details = {"host_usable_bytes": usable_host, "host_pending_bytes": pending,
                   "host_shared_cache_budget_bytes": self.shared_cache_budget_bytes,
                   "host_requested_bytes": request.host_bytes}
        if request.host_bytes > usable_host:
            return False, "host_memory", details
        if logical_gpu is not None:
            uuid = self.gpu_mapping[logical_gpu]
            gpu = snapshot.get("gpu", {}).get("devices", {}).get(uuid)
            if gpu is None:
                return False, "gpu_telemetry_unavailable", details
            reserve = max(self.gpu_reserve_bytes, int(gpu["total_bytes"] * self.reserve_fraction))
            pid_bytes = gpu.get("pid_bytes", {})
            pending = sum(max(0, item.request.gpu_bytes - pid_bytes.get(item.pid, pid_bytes.get(str(item.pid), 0)))
                          for item in self.active.values() if item.gpu_uuid == uuid)
            usable_gpu = gpu["free_bytes"] - reserve - pending
            details.update(gpu_uuid=uuid, gpu_usable_bytes=usable_gpu, gpu_pending_bytes=pending,
                           gpu_requested_bytes=request.gpu_bytes)
            if request.gpu_bytes > usable_gpu:
                return False, "gpu_memory", details
        return True, "admitted", details

    def choose(self, request, snapshot):
        options = list(self.gpu_mapping) if self.gpu_mapping else [None]
        candidates, rejected = [], []
        for logical in options:
            allowed, reason, details = self.admission(request, snapshot, logical)
            if allowed:
                candidates.append((details.get("gpu_usable_bytes", details["host_usable_bytes"]), logical, details))
            else:
                rejected.append({"logical_gpu": logical, "reason": reason, "details": details})
        if not candidates:
            return None, rejected
        _, logical, details = max(candidates, key=lambda item: item[0])
        return {"logical_gpu": logical, "device": "cpu" if logical is None else f"cuda:{logical}", "details": details}, rejected

    def reserve(self, job_id, pid, request, logical_gpu=None):
        if job_id in self.active:
            raise ValueError(f"Duplicate active job: {job_id}")
        reservation = Reservation(job_id, pid, request, self.gpu_mapping.get(logical_gpu), logical_gpu, time.time())
        self.active[job_id] = reservation
        self.record("reserved", reservation=asdict(reservation))

    def release(self, job_id):
        reservation = self.active.pop(job_id)
        self.record("released", job_id=job_id, pid=reservation.pid)

    def reduce_after_oom(self, reason="confirmed_oom"):
        self.max_parallel = max(1, self.max_parallel - 1)
        self.cpu_parallel_limit = min(self.cpu_parallel_limit, self.max_parallel)
        self.record("oom_concurrency_reduced", max_parallel=self.max_parallel, reason=reason)


def classify_oom(log_path):
    try:
        with Path(log_path).open("rb") as handle:
            handle.seek(max(0, Path(log_path).stat().st_size - 128 * 1024))
            tail = handle.read().decode("utf-8", "replace").lower()
    except OSError:
        return False
    return any(pattern in tail for pattern in ("cuda out of memory", "cudaoutofmemoryerror", "torch.outofmemoryerror", "cublas_status_alloc_failed", "defaultcpuallocator: can't allocate memory", "memoryerror:"))
