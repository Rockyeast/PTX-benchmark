#!/usr/bin/env python3

"""Fail closed unless CUDA device 0 is the B200 target required by this task."""

import argparse
import ctypes
import json
from pathlib import Path


CUDA_SUCCESS = 0
CC_MAJOR = 75
CC_MINOR = 76
EXPECTED_GPU_TYPE = "B200"
EXPECTED_SM = "100"


def check(result: int, operation: str) -> None:
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]

    check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

    name_buffer = ctypes.create_string_buffer(128)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    check(
        cuda.cuDeviceGetName(name_buffer, len(name_buffer), device),
        "cuDeviceGetName",
    )
    check(
        cuda.cuDeviceGetAttribute(ctypes.byref(major), CC_MAJOR, device),
        "cuDeviceGetAttribute(major)",
    )
    check(
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), CC_MINOR, device),
        "cuDeviceGetAttribute(minor)",
    )

    name = name_buffer.value.decode()
    actual_sm = f"{major.value}{minor.value}"
    payload = {
        "device_index": device.value,
        "device_name": name,
        "compute_capability": f"{major.value}.{minor.value}",
        "sm": actual_sm,
        "expected_gpu_type": EXPECTED_GPU_TYPE,
        "expected_sm": EXPECTED_SM,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if EXPECTED_GPU_TYPE.casefold() not in name.casefold():
        raise RuntimeError(
            f"expected GPU name containing {EXPECTED_GPU_TYPE!r}, got {name!r}"
        )
    if actual_sm != EXPECTED_SM:
        raise RuntimeError(f"expected sm_{EXPECTED_SM}, got sm_{actual_sm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
