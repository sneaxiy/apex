#pragma once

#include "stdint.h"
#include "stddef.h"
#include "cuda.h"
#include "cuda_runtime.h"

#ifdef __cplusplus
#include "dlfcn.h"
#include <cstdlib>
#include <stdexcept>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * All functions of the FMHALIB does not return the error flag and
 * does not throw any exception. If users want to check whether there
 * is any error, users can call this method after the FMHALIB function
 * is called.
 *
 * If there is no error, it would return nullptr.
 * If there is any error, it would return the error message.
 *
 * Note that if the empty string "" is returned, there is an error without
 * any error message instead of no errors. 
 *
 * Note that the error message is thread local, do not get the error message
 * in another thread.
 */
const char *fmhalib_error();

/**
 * qkv_ptr: FP16 Tensor with shape [total, 3, num_heads, head_size] 
 * cu_seqlens_ptr: INT32 Tensor with shape [batch_size + 1]
 * total: the total seqence length (not including padding) of the mini-batch   
 * num_heads: head number. 
 * head_size: must be 64
 * batch_size: batch size
 * p_dropout: dropout probability
 * max_seq_len: must be any of [128, 256, 384, 512]
 * is_training: whether to run train or inference
 * rnd_seed: the random seed
 * offset_ptr: the device pointer to generate the random seed. It can be NULL if is_device_rnd == false. 
 * rnd_offset: the random seed offset.
 * is_device_rnd: whether to generate the random seed in the device side. 
 * stream: the CUDA stream.
 *
 * ctx_ptr: output FP16 tensor with shape [total, num_heads, head_size]
 * s_ptr: output FP16 Tensor with shape [batch_size, num_heads, max_seq_len, max_seq_len]  
 */
void fmhalib_fwd(const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout,
                 const int max_seq_len,
                 const bool is_training,
                 const uint64_t rnd_seed,
                 const int64_t *offset_ptr,
                 const uint64_t rnd_offset,
                 bool is_device_rnd,
                 cudaStream_t stream,
                 void *ctx_ptr,
                 void *s_ptr);

/**
 * Almost the same with fmhalib_fwd except for that max_seq_len must be 512.
 */
void fmhalib_fwd_nl(const void *qkv_ptr,
                    const void *cu_seqlens_ptr,
                    const int total,
                    const int num_heads,
                    const int head_size,
                    const int batch_size,
                    const float p_dropout,
                    const int max_seq_len,
                    const bool is_training,
                    const uint64_t rnd_seed,
                    const int64_t *offset_ptr,
                    const uint64_t rnd_offset,
                    bool is_device_rnd,
                    cudaStream_t stream,
                    void *ctx_ptr,
                    void *s_ptr);

/**
 * dout_ptr: the gradient of the output `ctx_ptr` in fmhalib_fwd
 * qkv_ptr: same with the fmhalib_fwd 
 * cu_seqlens_ptr: same with the fmhalib_fwd
 * total: same with the fmhalib_fwd
 * num_heads: same with the fmhalib_fwd
 * head_size: same with the fmhalib_fwd
 * batch_size: same with the fmhalib_fwd
 * p_dropout: same with the fmhalib_fwd
 * max_seq_len: same with the fmhalib_fwd
 * stream: the CUDA stream
 *
 * softmax_ptr: the output `s_ptr` in fmhalib_fwd. Note that it may be overwritten inside the function!   
 * dqkv_ptr: the gradient of the input `qkv_ptr` in fmhalib_fwd
 */
void fmhalib_bwd(const void *dout_ptr,
                 const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout, 
                 const int max_seq_len,
                 cudaStream_t stream,
                 void *softmax_ptr,  // will be overwritten
                 void *dqkv_ptr);

/**
 * Almost the same with fmhalib_bwd except for that max_seq_len must be 512.
 * There is an extra FP16 output dkv_ptr with shape [total, num_chunks, 2, num_heads, head_size],
 * where num_chunks can be obtained by calling fmhalib_bwd_nl_num_chunks(batch_size). 
 */
void fmhalib_bwd_nl(const void *dout_ptr,
                    const void *qkv_ptr,
                    const void *cu_seqlens_ptr,
                    const int total,
                    const int num_heads,
                    const int head_size,
                    const int batch_size,
                    const float p_dropout,
                    const int max_seq_len,
                    cudaStream_t stream,
                    void *softmax_ptr,  // will be overwritten
                    void *dqkv_ptr,
		    void *dkv_ptr);

int fmhalib_random_increment(const int seq_len);

int fmhalib_bwd_nl_num_chunks(const int batch_size);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace fmhalib {

namespace {
inline void *OpenFMHALibHandle() {
  const char *env_var = "FMHALIB_PATH";
  const char *lib_path = std::getenv(env_var);
  if (!lib_path) {
    lib_path = "libfmha.so";
  }
  void *handle = dlopen(lib_path, RTLD_LAZY);
  if (!handle) {
    throw std::runtime_error(dlerror());
  }
  return handle; 
}	

inline void *GetFMHALibSymbol(const char *name) {
  static void *lib_handle = OpenFMHALibHandle();
  void *symbol = dlsym(lib_handle, name);
  if (!symbol) {
    throw std::runtime_error(dlerror());
  }
  return symbol;
}
}

#define _DEFINE_FMHALIB_DYNLOAD_FUNC(__sym_name, __func_name)       \
   namespace {                                                      \
   static auto DynLoad_##__sym_name() -> decltype(&::__sym_name) {  \
     using __FuncType = decltype(&::__sym_name);                    \
     static auto *__func = reinterpret_cast<__FuncType>(            \
                         GetFMHALibSymbol(#__sym_name));            \
     return __func;                                                 \
   }                                                                \
   }                                                                \
   template <typename ...__ARGS>                                    \
   inline auto __func_name(__ARGS... __args)                        \
               -> decltype(::__sym_name(__args...)) {               \
     return DynLoad_##__sym_name()(__args...);                      \
   }

_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_error, error);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_fwd, fwd);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_bwd, bwd);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_fwd_nl, fwd_nl);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_bwd_nl, bwd_nl);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_random_increment, random_increment);
_DEFINE_FMHALIB_DYNLOAD_FUNC(fmhalib_bwd_nl_num_chunks, bwd_nl_num_chunks);

} // namespace fmhalib
#endif
