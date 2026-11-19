#!/usr/bin/env python3
"""Report the identities of CUDA devices actually visible to this Python.

Run this module in a short-lived subprocess. Older PyTorch versions do not
expose ``get_device_properties(...).uuid``. For those versions, a one-byte
tensor establishes the selected PyTorch device's CUDA context, and the CUDA
driver identifies that exact context with ``cuCtxGetDevice`` and
``cuDeviceGetUuid_v2``. Neither a physical ordinal nor a GPU name is inferred.

The v2 UUID API is required: unlike its predecessor it identifies a subscribed
MIG compute instance separately from its parent GPU. This scheduler does not
support MIG; explicit MIG visibility or a MIG device name is reported as such.
Other identity errors leave an empty UUID and an explanatory note so the
caller can apply its independently verified, UUID-only visibility fallback.

This file imports torch only inside ``probe_cuda_devices``. Importing it from
the parent scheduler therefore never initializes a parent CUDA context.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import uuid as uuid_module


def canonical_gpu_uuid(value):
    """Normalize a full CUDA UUID, returning an empty string if unavailable."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        if len(value) == 16:
            return "GPU-" + str(uuid_module.UUID(bytes=bytes(value)))
        try:
            value = bytes(value).decode("ascii")
        except UnicodeDecodeError:
            return ""
    text = str(value).strip()
    if text.upper().startswith("GPU-"):
        text = text[4:]
    # UUID objects, canonical strings and 32-character hexadecimal strings are
    # supported. UUID prefixes/reprs/MIG identifiers are not complete IDs.
    if not re.fullmatch(r"(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", text):
        return ""
    return "GPU-" + str(uuid_module.UUID(text))


def _load_driver():
    return ctypes.CDLL("libcuda.so.1")


def _bind_context_uuid_api(driver):
    """Use stable driver ABI calls; never read a version-dependent C struct."""
    uuid_buffer_type = ctypes.c_ubyte * 16
    get_context_device = driver.cuCtxGetDevice
    get_context_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_context_device.restype = ctypes.c_int
    get_uuid = driver.cuDeviceGetUuid_v2
    get_uuid.argtypes = [ctypes.POINTER(uuid_buffer_type), ctypes.c_int]
    get_uuid.restype = ctypes.c_int

    def identity():
        device = ctypes.c_int()
        code = get_context_device(ctypes.byref(device))
        if code != 0:
            raise RuntimeError(f"cuCtxGetDevice returned CUDA driver error {code}")
        value = uuid_buffer_type()
        code = get_uuid(ctypes.byref(value), device.value)
        if code != 0:
            raise RuntimeError(f"cuDeviceGetUuid_v2 returned CUDA driver error {code}")
        return canonical_gpu_uuid(bytes(value))

    return identity


def probe_cuda_devices(torch_module=None, driver_loader=None):
    """Return JSON-safe CUDA visibility and per-logical-device identity data.

    Optional arguments are dependency injection points for GPU-free tests.
    Device identity failures are reported within ``identity_notes``. CUDA
    availability/enumeration errors propagate, preventing an accidental CPU
    fallback when the probe itself has failed.
    """
    if torch_module is None:
        import torch as torch_module
    if driver_loader is None:
        driver_loader = _load_driver

    cuda = torch_module.cuda
    result = {"available": bool(cuda.is_available()), "devices": [],
              "torch_version": str(getattr(torch_module, "__version__", "unknown")),
              "cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "unknown"))}
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    result["cuda_visible_devices"] = visibility
    explicit_mig = any(token.strip().upper().startswith("MIG-")
                       for token in (visibility or "").split(","))
    result["unsupported_mig"] = explicit_mig
    if not result["available"]:
        return result

    context_identity = None
    driver_failure = None
    for logical in range(cuda.device_count()):
        record = {"logical": logical, "uuid": "", "identity_method": "unavailable",
                  "unsupported_mig": explicit_mig, "identity_notes": []}
        raw_uuid = None
        try:
            properties = cuda.get_device_properties(logical)
            record["name"] = str(getattr(properties, "name", "unknown"))
            if re.search(r"\bMIG\b", record["name"], flags=re.IGNORECASE):
                record["unsupported_mig"] = True
                result["unsupported_mig"] = True
            if getattr(properties, "total_memory", None) is not None:
                record["total_memory"] = int(properties.total_memory)
            raw_uuid = getattr(properties, "uuid", None)
            if str(raw_uuid).strip().upper().startswith("MIG-"):
                record["unsupported_mig"] = True
                result["unsupported_mig"] = True
            record["uuid"] = canonical_gpu_uuid(raw_uuid)
        except Exception as error:
            record["identity_notes"].append(f"torch_device_properties: {type(error).__name__}: {error}")

        if record["uuid"]:
            record["identity_method"] = "torch_device_properties_uuid"
        elif not record["unsupported_mig"]:
            if raw_uuid is not None and str(raw_uuid).strip():
                record["identity_notes"].append("PyTorch exposed no full, parseable device UUID")
            if context_identity is None and driver_failure is None:
                try:
                    context_identity = _bind_context_uuid_api(driver_loader())
                except Exception as error:
                    driver_failure = f"{type(error).__name__}: {error}"
            if driver_failure is not None:
                record["identity_notes"].append("cuda_current_context_uuid_v2: " + driver_failure)
            else:
                try:
                    with cuda.device(logical):
                        # The driver query sees the context used by this exact
                        # PyTorch logical device even after visibility reorders.
                        context_marker = torch_module.empty((1,), device=f"cuda:{logical}", dtype=torch_module.uint8)
                        try:
                            record["uuid"] = context_identity()
                        finally:
                            del context_marker
                    if record["uuid"]:
                        record["identity_method"] = "cuda_current_context_uuid_v2"
                except Exception as error:
                    record["identity_notes"].append(f"cuda_current_context_uuid_v2: {type(error).__name__}: {error}")
        if record["unsupported_mig"]:
            record["identity_notes"].append("MIG requires instance-level telemetry; automatic shared-GPU scheduling is unsupported")
        result["devices"].append(record)
    return result


def main():
    try:
        print(json.dumps(probe_cuda_devices(), ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"probe_error": f"{type(error).__name__}: {error}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
