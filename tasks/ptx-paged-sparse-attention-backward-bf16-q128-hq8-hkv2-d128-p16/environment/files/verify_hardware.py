#!/usr/bin/env python3
import argparse
import ctypes
import json
from pathlib import Path

def check(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    name_buffer = ctypes.create_string_buffer(128)
    major, minor = ctypes.c_int(), ctypes.c_int()
    check(cuda.cuDeviceGetName(name_buffer, len(name_buffer), device), "cuDeviceGetName")
    check(cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device), "compute capability major")
    check(cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device), "compute capability minor")
    name = name_buffer.value.decode()
    payload = {"device_index": device.value, "device_name": name,
               "compute_capability": f"{major.value}.{minor.value}",
               "sm": f"{major.value}{minor.value}", "expected_gpu_type": "B200", "expected_sm": "100"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if "b200" not in name.casefold() or (major.value, minor.value) != (10, 0):
        raise RuntimeError(f"expected NVIDIA B200 sm_100, got {name} sm_{major.value}{minor.value}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
