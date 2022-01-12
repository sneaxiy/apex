#pragma once

#include "stdint.h"
#include "stddef.h"
#include "cuda.h"
#include "cuda_runtime.h"

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
 * This function may return -1 if seq_len is invalid.
 */
int fmhalib_seq_len(const int seq_len);

/**
 * qkv_ptr: FP16 Tensor with shape [total, 3, num_heads, head_size] 
 * cu_seqlens_ptr: INT32 Tensor with shape [batch_size + 1]
 * total: the total seqence length (not including padding) of the mini-batch   
 * num_heads: head number 
 * head_size: must be 64
 * batch_size: batch size
 * p_dropout: dropout probability
 * max_seq_len: must be <= 512
 * is_training: whether to run train or inference
 * rnd_seed: the random seed
 * offset_ptr: the device pointer to generate the random seed. It can be NULL if is_device_rnd == false. 
 * rnd_offset: the random seed offset.
 * is_device_rnd: whether to generate the random seed in the device side. 
 * stream: the CUDA stream.
 *
 * ctx_ptr: output FP16 tensor with shape [total, num_heads, head_size]
 * s_ptr: output FP16 Tensor with shape [batch_size, num_heads, seq_len, seq_len], where seq_len can be obtained by calling `fmhalib_seq_len(max_seq_len)` 
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
                 const bool is_device_rnd,
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
                    const bool is_device_rnd,
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

/**
 * Set the random seed to the inner seed manager. 
 *
 * This function may call `cudaMalloc` when it is
 * called at the first time.
 *
 * This function may report error when `cudaMalloc`
 * fails. Please get the error by calling
 * `fmhalib_error()`.   
 */
void fmhalib_random_seed(const uint64_t rnd_seed); 

/**
 * Increase the state of the inner seed manager by
 * `increment`. The `increment` can be obtained by
 * calling `fmhalib_random_increment(seq_len)`.  
 *
 * This function may report error when any of
 * `rnd_seed`, `offset_ptr`, `rnd_offset` and
 * `is_device_rnd` is NULL. Please get the error by 
 * calling `fmhalib_error()`.
 *
 * TODO(zengjinle): support is_device_rnd = true.
 */
void fmhalib_random_state(const uint64_t increment,
                          uint64_t *rnd_seed,
                          int64_t **offset_ptr,
                          uint64_t *rnd_offset,
                          bool *is_device_rnd);

// This function never reports error 
int fmhalib_random_increment(const int seq_len);

// This function never reports error 
int fmhalib_bwd_nl_num_chunks(const int batch_size);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
namespace fmhalib {

#define _DEFINE_CXX_FMHALIB_DYNLOAD_FUNC(__func_name)     \
   extern decltype(&fmhalib_##__func_name)                \
       DynLoad_##fmhalib_##__func_name();                 \
   template <typename ...__ARGS>                          \
   inline auto __func_name(__ARGS... __args)              \
       -> decltype(::fmhalib_##__func_name(__args...)) {  \
     return DynLoad_##fmhalib_##__func_name()(__args...); \
   }

#define _CXX_FMHALIB_FOR_EACH_FUNC(__macro) \
   __macro(error);                          \
   __macro(seq_len);                        \
   __macro(fwd);                            \
   __macro(bwd);                            \
   __macro(fwd_nl);                         \
   __macro(bwd_nl);                         \
   __macro(random_seed);                    \
   __macro(random_state);                   \
   __macro(random_increment);               \
   __macro(bwd_nl_num_chunks)

_CXX_FMHALIB_FOR_EACH_FUNC(_DEFINE_CXX_FMHALIB_DYNLOAD_FUNC);

} // namespace fmhalib
#endif
