#!/usr/bin/env python3
"""Correctness-gated verifier for fixed paged GQA decode on B200."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import struct
from array import array
from pathlib import Path

ENTRY = b"paged_decode_fwd_bf16_hq32_hkv8_d128_p16"
HQ, HKV, D = 32, 8, 128
PAGE_SIZE, LOGICAL_PAGES, PHYSICAL_PAGES = 16, 256, 320
MAX_KV = PAGE_SIZE * LOGICAL_PAGES
CACHE_ELEMENTS = PHYSICAL_PAGES * PAGE_SIZE * HKV * D
BLOCK_SIZE = 128
GUARD_BYTES = 256
MAX_PTX_BYTES = 1 << 20
ABS_TOL, REL_TOL = 2e-2, 2e-2
OUTPUT_SENTINEL = 0x7FC0
POISON = 0x4200  # BF16 32.0
CORRECTNESS_CASES = [0, 1, 15, 16, 17, 97, 512, 4095, 4096]
PERFORMANCE_LENGTHS = [128, 1024, 4096]


class CudaError(RuntimeError): pass
class CandidateRejected(Exception): pass


def candidate_call(operation, function, *args):
    try: return function(*args)
    except CandidateRejected: raise
    except Exception as error: raise CandidateRejected(f"{operation}: {error}") from error


class CudaDriver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._signatures()
        self.device, self.context = ctypes.c_int(), ctypes.c_void_p()
        self.allocations = set()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device), "cuCtxCreate")

    def _signatures(self):
        l = self.lib
        l.cuInit.argtypes=[ctypes.c_uint]; l.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
        l.cuDeviceGetName.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
        l.cuCtxCreate_v2.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint,ctypes.c_int]
        l.cuCtxSynchronize.argtypes=[]; l.cuCtxDestroy_v2.argtypes=[ctypes.c_void_p]
        l.cuModuleLoad.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_char_p]
        l.cuModuleUnload.argtypes=[ctypes.c_void_p]
        l.cuModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
        l.cuMemAlloc_v2.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]
        l.cuMemFree_v2.argtypes=[ctypes.c_uint64]
        l.cuMemcpyHtoD_v2.argtypes=[ctypes.c_uint64,ctypes.c_void_p,ctypes.c_size_t]
        l.cuMemcpyDtoH_v2.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_size_t]
        l.cuLaunchKernel.argtypes=[ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,
            ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
        l.cuEventCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint]
        l.cuEventRecord.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        l.cuEventSynchronize.argtypes=[ctypes.c_void_p]
        l.cuEventElapsedTime.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_void_p,ctypes.c_void_p]
        l.cuEventDestroy_v2.argtypes=[ctypes.c_void_p]
        l.cuGetErrorName.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]

    def error_name(self, result):
        name=ctypes.c_char_p()
        return name.value.decode() if self.lib.cuGetErrorName(result,ctypes.byref(name))==0 and name.value else "UNKNOWN"
    def check(self,result,operation):
        if result: raise CudaError(f"{operation} failed: {self.error_name(result)} ({result})")
    def device_name(self):
        b=ctypes.create_string_buffer(256); self.check(self.lib.cuDeviceGetName(b,len(b),self.device),"cuDeviceGetName"); return b.value.decode()
    def load_function(self,path):
        module,function=ctypes.c_void_p(),ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module),str(path).encode()),f"cuModuleLoad({path})")
        try: self.check(self.lib.cuModuleGetFunction(ctypes.byref(function),module,ENTRY),f"cuModuleGetFunction({ENTRY.decode()})")
        except Exception: self.lib.cuModuleUnload(module); raise
        return module,function
    def alloc(self,size):
        p=ctypes.c_uint64(); self.check(self.lib.cuMemAlloc_v2(ctypes.byref(p),size),"cuMemAlloc"); self.allocations.add(p.value); return p.value
    def free(self,p):
        if p in self.allocations: self.check(self.lib.cuMemFree_v2(p),"cuMemFree"); self.allocations.remove(p)
    def copy_to_device(self,p,data):
        h=(ctypes.c_ubyte*len(data)).from_buffer_copy(data); self.check(self.lib.cuMemcpyHtoD_v2(p,h,len(data)),"cuMemcpyHtoD")
    def copy_from_device(self,p,size):
        h=(ctypes.c_ubyte*size)(); self.check(self.lib.cuMemcpyDtoH_v2(h,p,size),"cuMemcpyDtoH"); return bytes(h)
    def launch(self,function,q,k,v,pages,out,kv_len):
        raw=[ctypes.c_uint64(q),ctypes.c_uint64(k),ctypes.c_uint64(v),ctypes.c_uint64(pages),ctypes.c_uint64(out),ctypes.c_uint32(kv_len)]
        args=(ctypes.c_void_p*len(raw))(*(ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in raw))
        self.check(self.lib.cuLaunchKernel(function,HQ,1,1,BLOCK_SIZE,1,1,0,None,args,None),"cuLaunchKernel")
    def synchronize(self): self.check(self.lib.cuCtxSynchronize(),"cuCtxSynchronize")
    def measure_ms(self,function,pointers,kv_len,iterations):
        a,b=ctypes.c_void_p(),ctypes.c_void_p(); self.check(self.lib.cuEventCreate(ctypes.byref(a),0),"cuEventCreate") ; self.check(self.lib.cuEventCreate(ctypes.byref(b),0),"cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(a,None),"cuEventRecord")
            for _ in range(iterations): self.launch(function,*pointers,kv_len)
            self.check(self.lib.cuEventRecord(b,None),"cuEventRecord"); self.check(self.lib.cuEventSynchronize(b),"cuEventSynchronize")
            ms=ctypes.c_float(); self.check(self.lib.cuEventElapsedTime(ctypes.byref(ms),a,b),"cuEventElapsedTime"); return float(ms.value)/iterations
        finally:
            if b.value: self.lib.cuEventDestroy_v2(b)
            if a.value: self.lib.cuEventDestroy_v2(a)
    def close(self):
        for p in tuple(self.allocations): self.lib.cuMemFree_v2(p)
        if self.context.value: self.lib.cuCtxDestroy_v2(self.context)


def float_to_bf16(value):
    bits=struct.unpack("<I",struct.pack("<f",value))[0]
    if (bits&0x7F800000)==0x7F800000 and bits&0x007FFFFF: return 0x7FC0
    return ((bits+0x7FFF+((bits>>16)&1))>>16)&0xFFFF
def bf16_to_float(value): return struct.unpack("<f",struct.pack("<I",value<<16))[0]


class Truth:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path)); p16=ctypes.POINTER(ctypes.c_uint16); p32=ctypes.POINTER(ctypes.c_uint32)
        self.lib.paged_decode_truth.argtypes=[p16,p16,p16,p32,p16,ctypes.c_uint32]; self.lib.paged_decode_truth.restype=ctypes.c_int
    def compute(self,q,k,v,pages,kv_len):
        out=array("H",[0])*(HQ*D)
        qb=(ctypes.c_uint16*len(q)).from_buffer(q); kb=(ctypes.c_uint16*len(k)).from_buffer(k); vb=(ctypes.c_uint16*len(v)).from_buffer(v)
        pb=(ctypes.c_uint32*len(pages)).from_buffer(pages); ob=(ctypes.c_uint16*len(out)).from_buffer(out)
        result=self.lib.paged_decode_truth(qb,kb,vb,pb,ob,kv_len)
        if result: raise RuntimeError(f"CPU truth returned {result}")
        return out


def page_map(): return array("I", ((i*73+19)%PHYSICAL_PAGES for i in range(LOGICAL_PAGES)))


def make_case(kv_len):
    q=array("H",[0])*(HQ*D); k=array("H",[POISON])*CACHE_ELEMENTS; v=array("H",[POISON])*CACHE_ELEMENTS; pages=page_map()
    for h in range(HQ):
        for d in range(D): q[h*D+d]=float_to_bf16(((h*13+d*7)%31-15)/16.0)
    for token in range(kv_len):
        physical=pages[token//PAGE_SIZE]; offset=token%PAGE_SIZE
        for hk in range(HKV):
            base=(((physical*PAGE_SIZE+offset)*HKV+hk)*D)
            for d in range(D):
                kval=((token*17+hk*11+d*5)%37-18)/12.0
                vval=((token*19+hk*23+d*3)%41-20)/10.0+hk*0.03125
                if kv_len>=512 and token==(hk*499+37)%kv_len: kval*=4.0
                k[base+d]=float_to_bf16(kval); v[base+d]=float_to_bf16(vval)
    return q,k,v,pages


def guarded(payload,byte): return bytearray([byte])*GUARD_BYTES+payload+bytearray([byte])*GUARD_BYTES
def assert_guards(payload,byte,label):
    g=bytes([byte])*GUARD_BYTES
    if payload[:GUARD_BYTES]!=g or payload[-GUARD_BYTES:]!=g: raise CandidateRejected(f"{label}: memory guard changed")
def assert_output(payload,expected,label):
    actual=array("H"); actual.frombytes(payload)
    if len(actual)!=len(expected): raise RuntimeError(f"{label}: output size mismatch")
    for i,(gb,wb) in enumerate(zip(actual,expected)):
        got,want=bf16_to_float(gb),bf16_to_float(wb)
        if not math.isfinite(got) or not math.isclose(got,want,rel_tol=REL_TOL,abs_tol=ABS_TOL):
            raise CandidateRejected(f"{label}: wrong output index={i}: expected {want}, got {got}")


def run_case(cuda,function,truth,kv_len,label):
    q,k,v,pages=make_case(kv_len); expected=truth.compute(q,k,v,pages,kv_len)
    payloads={"q":q.tobytes(),"k":k.tobytes(),"v":v.tobytes(),"pages":pages.tobytes(),"out":(array("H",[OUTPUT_SENTINEL])*len(expected)).tobytes()}
    bytes_by_name={"q":0xA1,"k":0xB2,"v":0xC3,"pages":0xD4,"out":0xE5}
    hosts={n:guarded(data,bytes_by_name[n]) for n,data in payloads.items()}; devices={n:cuda.alloc(len(data)) for n,data in hosts.items()}; after={}
    try:
        for n,data in hosts.items(): cuda.copy_to_device(devices[n],data)
        candidate_call(f"{label} launch kv_len={kv_len}",cuda.launch,function,*(devices[n]+GUARD_BYTES for n in ("q","k","v","pages","out")),kv_len)
        candidate_call(f"{label} sync kv_len={kv_len}",cuda.synchronize)
        for n,data in hosts.items(): after[n]=candidate_call(f"{label} copy {n}",cuda.copy_from_device,devices[n],len(data))
    finally:
        for p in reversed(tuple(devices.values())):
            try: cuda.free(p)
            except Exception: pass
    for n in ("q","k","v","pages"):
        if after[n]!=bytes(hosts[n]): raise CandidateRejected(f"{label}: {n} modified at kv_len={kv_len}")
    assert_guards(after["out"],0xE5,f"{label} output kv_len={kv_len}")
    assert_output(after["out"][GUARD_BYTES:GUARD_BYTES+len(expected)*2],expected,f"{label} kv_len={kv_len}")


def validate_function(cuda,function,truth,label):
    for kv_len in CORRECTNESS_CASES: run_case(cuda,function,truth,kv_len,label)


def dispersion(values):
    median=statistics.median(values); return math.inf if median==0 else (max(values)-min(values))/median


def benchmark_shape(cuda,candidate,reference,truth,kv_len):
    q,k,v,pages=make_case(kv_len); expected=truth.compute(q,k,v,pages,kv_len)
    buffers=[q.tobytes(),k.tobytes(),v.tobytes(),pages.tobytes(),(array("H",[OUTPUT_SENTINEL])*len(expected)).tobytes()]
    pointers=tuple(cuda.alloc(len(x)) for x in buffers)
    try:
        for p,data in zip(pointers,buffers): cuda.copy_to_device(p,data)
        for label,function in (("candidate",candidate),("reference",reference)):
            try: cuda.launch(function,*pointers,kv_len); cuda.synchronize(); assert_output(cuda.copy_from_device(pointers[4],len(expected)*2),expected,f"{label} performance kv_len={kv_len}")
            except CandidateRejected:
                if label=="reference": raise RuntimeError("fixed Reference performance correctness failed")
                raise
        for _ in range(5): cuda.launch(candidate,*pointers,kv_len); cuda.launch(reference,*pointers,kv_len)
        cuda.synchronize(); cp=candidate_call("Candidate timing probe",cuda.measure_ms,candidate,pointers,kv_len,3); rp=cuda.measure_ms(reference,pointers,kv_len,3)
        iterations=max(3,min(1000,math.ceil(50.0/max(cp,rp,1e-6)))); times={"candidate":[],"reference":[]}
        for batch in range(5):
            order=(("candidate",candidate),("reference",reference)) if batch%2==0 else (("reference",reference),("candidate",candidate))
            for name,function in order:
                elapsed=candidate_call("Candidate timing",cuda.measure_ms,function,pointers,kv_len,iterations) if name=="candidate" else cuda.measure_ms(function,pointers,kv_len,iterations)
                times[name].append(elapsed)
        cm,rm=statistics.median(times["candidate"]),statistics.median(times["reference"]); ratio=rm/cm; flops=4*HQ*kv_len*D
        return {"kv_len":kv_len,"iterations_per_batch":iterations,"candidate_ms_per_launch":times["candidate"],"reference_ms_per_launch":times["reference"],"candidate_median_ms":cm,"reference_median_ms":rm,"speedup_vs_reference":ratio,"candidate_logical_tflops":flops/(cm*1e9),"reference_logical_tflops":flops/(rm*1e9),"candidate_dispersion":dispersion(times["candidate"]),"reference_dispersion":dispersion(times["reference"])}
    finally:
        for p in reversed(pointers):
            try: cuda.free(p)
            except Exception: pass


def validate_ptx(path,candidate):
    role="Candidate" if candidate else "fixed Reference"; error=CandidateRejected if candidate else RuntimeError
    if not path.is_file(): raise error(f"missing {role} PTX: {path}")
    if path.stat().st_size==0 or path.stat().st_size>MAX_PTX_BYTES: raise error(f"{role} PTX has invalid size")
    try: source=path.read_text()
    except UnicodeDecodeError as exc: raise error(f"{role} PTX is not UTF-8") from exc
    if ".entry" not in source or ENTRY.decode() not in source: raise error(f"{role} PTX missing entry {ENTRY.decode()}")


def parse_args():
    p=argparse.ArgumentParser()
    for name in ("candidate","reference","truth","log","reward"): p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--correctness-only",action="store_true"); return p.parse_args()
def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); temp.replace(path)


def run(args):
    validate_ptx(args.reference,False); validate_ptx(args.candidate,True); truth=Truth(args.truth); cuda=CudaDriver(); cm,rm=ctypes.c_void_p(),ctypes.c_void_p()
    try:
        rm,reference=cuda.load_function(args.reference)
        try: validate_function(cuda,reference,truth,"reference")
        except CandidateRejected as e: raise RuntimeError(f"fixed Reference correctness failed: {e}") from e
        cm,candidate=candidate_call("Candidate module load",cuda.load_function,args.candidate)
        candidate_call("Candidate correctness",validate_function,cuda,candidate,truth,"candidate")
        metrics={"status":"correctness_passed","gpu":cuda.device_name(),"entry":ENTRY.decode(),"correctness_kv_lengths":CORRECTNESS_CASES,"atol":ABS_TOL,"rtol":REL_TOL}
        if args.correctness_only: reward={"reward":0.0,"correctness":1.0,"performance":0.0,"speedup_vs_reference":0.0}
        else:
            timing=[benchmark_shape(cuda,candidate,reference,truth,n) for n in PERFORMANCE_LENGTHS]; ratios=[x["speedup_vs_reference"] for x in timing]; raw=math.exp(sum(math.log(x) for x in ratios)/len(ratios)); performance=min(1.0,max(0.0,raw)); metrics.update(status="passed",timing=timing,raw_speedup_vs_reference=raw); reward={"reward":performance,"correctness":1.0,"performance":performance,"speedup_vs_reference":raw}
        return metrics,reward
    finally:
        if cm.value: cuda.lib.cuModuleUnload(cm)
        if rm.value: cuda.lib.cuModuleUnload(rm)
        cuda.close()


def main():
    args=parse_args(); reward={"reward":0.0,"correctness":0.0,"performance":0.0,"speedup_vs_reference":0.0}; metrics={"status":"failed","entry":ENTRY.decode()}
    try: metrics,reward=run(args)
    except CandidateRejected as e: metrics.update(status="candidate_rejected",error_type=type(e).__name__,error=str(e)); print(f"PTX verification failed: {e}"); write_json(args.log,metrics); write_json(args.reward,reward); return 1
    except Exception as e: metrics.update(status="infrastructure_error",error_type=type(e).__name__,error=str(e)); print(f"PTX verifier infrastructure failure: {e}"); write_json(args.log,metrics); write_json(args.reward,reward); return 2
    write_json(args.log,metrics); write_json(args.reward,reward); return 0
if __name__=="__main__": raise SystemExit(main())
