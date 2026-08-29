/*
 * Copyright 2026 INT21 AI
 * SPDX-License-Identifier: MIT
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>

#include <array>
#include <cstdint>
#include <exception>
#include <limits>

namespace {

extern "C" cudaError_t flash_kda_decode_sm100a_launch_t1_split16(
    void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t1_split8(
    void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t2_split4(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t3_split4(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, int, float, float,
    cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t4_split2(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t5_split1(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t6_split1(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, int, float, cudaStream_t);

void* decode_pointer(unsigned long long value) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(value));
}

bool unpack_t1_tensors(
    std::array<PyObject*, 7> const& objects, int requested_device,
    std::array<at::Tensor, 7>* tensors, std::array<void*, 7>* pointers) {
  static constexpr std::array<char const*, 7> names = {
      "q", "k", "v", "g", "beta", "state", "out"};
  for (size_t index = 0; index < objects.size(); ++index) {
    try {
      if (!THPVariable_Check(objects[index])) {
        if (!PyErr_Occurred()) {
          PyErr_Format(PyExc_TypeError, "_decode_run_t1: %s must be a torch.Tensor",
                       names[index]);
        }
        return false;
      }
      (*tensors)[index] = THPVariable_Unpack(objects[index]);
      c10::Device const tensor_device = (*tensors)[index].device();
      if (!tensor_device.is_cuda()) {
        PyErr_Format(PyExc_ValueError,
                     "_decode_run_t1: %s must be a CUDA tensor on device %d (got %s)",
                     names[index], requested_device, tensor_device.str().c_str());
        return false;
      }
      if (tensor_device.index() != requested_device) {
        PyErr_Format(PyExc_ValueError,
                     "_decode_run_t1: %s is on CUDA device %d, expected device %d",
                     names[index], static_cast<int>(tensor_device.index()), requested_device);
        return false;
      }
      (*pointers)[index] = (*tensors)[index].data_ptr();
    } catch (std::exception const& error) {
      if (!PyErr_Occurred()) {
        PyErr_Format(PyExc_RuntimeError, "_decode_run_t1: failed to unpack %s: %s",
                     names[index], error.what());
      }
      return false;
    }
  }
  return true;
}

bool current_decode_stream(int device, cudaStream_t* stream) {
  try {
    *stream = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(device)).stream();
    return true;
  } catch (std::exception const& error) {
    PyErr_Format(PyExc_RuntimeError, "getCurrentCUDAStream(%d) failed: %s", device, error.what());
    return false;
  }
}

bool select_decode_device(int device) {
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError, "cudaGetDevice() failed before selecting device %d: %s", device,
                 cudaGetErrorString(status));
    return false;
  }
  if (current_device == device) return true;
  status = cudaSetDevice(device);
  if (status != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError, "cudaSetDevice(%d) failed: %s", device,
                 cudaGetErrorString(status));
    return false;
  }
  return true;
}

PyObject* check_decode_launch(cudaError_t status, char const* variant) {
  if (status != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError, "%s launch failed: %s", variant,
                 cudaGetErrorString(status));
    return nullptr;
  }
  Py_RETURN_NONE;
}

bool parse_decode_int(PyObject* object, int* value) {
  long const converted = PyLong_AsLong(object);
  if (converted == -1 && PyErr_Occurred()) return false;
  if (converted < std::numeric_limits<int>::min()) {
    PyErr_SetString(PyExc_OverflowError, "signed integer is less than minimum");
    return false;
  }
  if (converted > std::numeric_limits<int>::max()) {
    PyErr_SetString(PyExc_OverflowError, "signed integer is greater than maximum");
    return false;
  }
  *value = static_cast<int>(converted);
  return true;
}

bool parse_decode_double(PyObject* object, double* value) {
  double const converted = PyFloat_AsDouble(object);
  if (converted == -1.0 && PyErr_Occurred()) return false;
  *value = converted;
  return true;
}

PyObject* run_decode_t1(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 10) {
    PyErr_Format(PyExc_TypeError, "function takes exactly 10 arguments (%zd given)", nargs);
    return nullptr;
  }
  std::array<PyObject*, 7> objects = {
      args[0], args[1], args[2], args[3], args[4], args[5], args[6]};
  int sequences, device;
  double scale;
  if (!parse_decode_int(args[7], &sequences) || !parse_decode_int(args[8], &device) ||
      !parse_decode_double(args[9], &scale)) {
    return nullptr;
  }
  std::array<at::Tensor, 7> tensors;
  std::array<void*, 7> pointers;
  if (!unpack_t1_tensors(objects, device, &tensors, &pointers)) return nullptr;
  if (!select_decode_device(device)) return nullptr;
  cudaStream_t stream;
  if (!current_decode_stream(device, &stream)) return nullptr;
  float const scale_f = static_cast<float>(scale);
  if (sequences <= 16) {
    return check_decode_launch(
        flash_kda_decode_sm100a_launch_t1_split16(
            pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[5],
            pointers[6], sequences, scale_f, stream),
        "d128_t1_cached_direct_split16");
  }
  return check_decode_launch(
      flash_kda_decode_sm100a_launch_t1_split8(
          pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[5],
          pointers[6], sequences, scale_f, stream),
      "d128_t1_cached_direct_split8");
}

PyObject* run_decode_cached(PyObject*, PyObject* args) {
  unsigned long long q, k, v, g, beta, state, out, state_indices, accepted;
  int sequences, tokens, device;
  double scale;
  if (!PyArg_ParseTuple(args, "KKKKKKKKKiiid", &q, &k, &v, &g, &beta, &state, &out,
                        &state_indices, &accepted, &sequences, &tokens, &device, &scale)) {
    return nullptr;
  }
  if (!select_decode_device(device)) return nullptr;
  cudaStream_t stream;
  if (!current_decode_stream(device, &stream)) return nullptr;
  float const scale_f = static_cast<float>(scale);
  cudaError_t status = cudaErrorInvalidValue;
  char const* variant = "invalid_cached_variant";
  if (tokens == 2) {
    status = flash_kda_decode_sm100a_launch_t2_split4(
        decode_pointer(q), decode_pointer(k), decode_pointer(v), decode_pointer(g),
        decode_pointer(beta), decode_pointer(state), decode_pointer(out),
        decode_pointer(state_indices), decode_pointer(accepted), sequences, scale_f,
        stream);
    variant = "d128_t2_cached_split4";
  } else if (tokens == 4) {
    status = flash_kda_decode_sm100a_launch_t4_split2(
        decode_pointer(q), decode_pointer(k), decode_pointer(v), decode_pointer(g),
        decode_pointer(beta), decode_pointer(state), decode_pointer(out),
        decode_pointer(state_indices), decode_pointer(accepted), sequences, scale_f,
        stream);
    variant = "d128_t4_cached_split2";
  } else if (tokens == 5) {
    status = flash_kda_decode_sm100a_launch_t5_split1(
        decode_pointer(q), decode_pointer(k), decode_pointer(v), decode_pointer(g),
        decode_pointer(beta), decode_pointer(state), decode_pointer(out),
        decode_pointer(state_indices), decode_pointer(accepted), sequences, scale_f,
        stream);
    variant = "d128_t5_cached_gram_split1";
  } else if (tokens == 6) {
    status = flash_kda_decode_sm100a_launch_t6_split1(
        decode_pointer(q), decode_pointer(k), decode_pointer(v), decode_pointer(g),
        decode_pointer(beta), decode_pointer(state), decode_pointer(out),
        decode_pointer(state_indices), decode_pointer(accepted), sequences, scale_f,
        stream);
    variant = "d128_t6_cached_gram_split1";
  }
  return check_decode_launch(status, variant);
}

PyObject* run_decode_t3(PyObject*, PyObject* args) {
  unsigned long long q, k, v, g, beta, a_log, dt_bias, state, out, state_indices, accepted;
  int sequences, device;
  double scale, lower_bound;
  if (!PyArg_ParseTuple(args, "KKKKKKKKKKKiidd", &q, &k, &v, &g, &beta, &a_log, &dt_bias,
                        &state, &out, &state_indices, &accepted, &sequences, &device, &scale,
                        &lower_bound)) {
    return nullptr;
  }
  if (!select_decode_device(device)) return nullptr;
  cudaStream_t stream;
  if (!current_decode_stream(device, &stream)) return nullptr;
  return check_decode_launch(
      flash_kda_decode_sm100a_launch_t3_split4(
          decode_pointer(q), decode_pointer(k), decode_pointer(v), decode_pointer(g),
          decode_pointer(beta), decode_pointer(a_log), decode_pointer(dt_bias),
          decode_pointer(state), decode_pointer(out), decode_pointer(state_indices),
          decode_pointer(accepted), sequences, static_cast<float>(scale),
          static_cast<float>(lower_bound), stream),
      "d128_t3_lower_bound_split4");
}

static PyMethodDef decode_methods[] = {
    {"_decode_run_t1", _PyCFunction_CAST(run_decode_t1), METH_FASTCALL,
     "Launch the private T=1 cached SM100a specialization."},
    {"_decode_run_cached", run_decode_cached, METH_VARARGS,
     "Launch a private cached-gate SM100a specialization."},
    {"_decode_run_t3", run_decode_t3, METH_VARARGS,
     "Launch the private T=3 lower-bound SM100a specialization."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

extern "C" __attribute__((visibility("default"))) int flash_kda_register_decode_sm100a(
    PyObject* module) {
  return PyModule_AddFunctions(module, decode_methods);
}
