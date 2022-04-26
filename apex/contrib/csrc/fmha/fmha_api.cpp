/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include "fmha.h"
#include <cstring>
#include <string>
#include <exception>
#include <stdexcept>
#include <mutex>
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "fmhalib.h"
#include "dlfcn.h"

using SeedIncFuncPtr = void (*)(uint64_t, uint64_t *, const int64_t **, uint64_t*, bool*);

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
		__FILE__ + ":" +                         \
		::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)


thread_local std::unique_ptr<char[]> fmhalib_err_msg;  

#ifdef __cplusplus
extern "C" {
#endif

const char *fmhalib_version() {
   return "0.1";
}

static void fmhalib_set_error(const char *msg) {
  if (msg == nullptr || *msg == '\0') {
    msg = "unknown error";
  }

  auto n = strlen(msg);
  std::unique_ptr<char[]> new_err_msg(new char[n+1]);
  std::strcpy(new_err_msg.get(), msg);
  fmhalib_err_msg = std::move(new_err_msg);
}

const char *fmhalib_error() {
  return fmhalib_err_msg.get();
}

#define FMHALIB_BEGIN_FUNC try {
#define FMHALIB_END_FUNC } catch (::std::exception &__e) { fmhalib_set_error(__e.what()); } catch (...) { fmhalib_set_error(nullptr); }

static void set_params(Fused_multihead_attention_fprop_params &params,
                       // sizes
                       const size_t b,
                       const size_t s,
                       const size_t h,
                       const size_t d,
                       // device pointers
                       void *qkv_packed_d,
                       void *cu_seqlens_d,
                       void *o_packed_d,
                       void *s_d,
                       float p_dropout) {

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = qkv_packed_d;
    params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    const float scale_bmm1 = 1.f / sqrtf(d);
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    set_alpha(params.scale_bmm1, scale_bmm1, data_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.rp_dropout = 1.f / params.p_dropout;
    ASSERT_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);
}

static int fmhalib_bwd_nl_num_chunks(const int batch_size) {
    int num_chunks = 2;
    if( batch_size == 1 ) {
        num_chunks = 4;
    }else if( batch_size == 2 ) {
        num_chunks = 3;
    }
    return num_chunks;
}

int fmhalib_seq_len(const int max_seq_len) {
    if (max_seq_len < 0) {
      return -1;
    }

    const int seq_lens[] = {128, 256, 384, 512};
    constexpr int n = sizeof(seq_lens) / sizeof(seq_lens[0]);
#pragma unroll n
    for (int i = 0; i < n; ++i) {
      if (max_seq_len <= seq_lens[i]) {
	return seq_lens[i];
      }
    }
    return -1;
}

static void SetPhiloxCudaState(at::PhiloxCudaState *state, SeedIncFuncPtr seed_inc_func, uint64_t increment) {
    uint64_t rnd_seed;
    const int64_t *offset_ptr;
    uint64_t rnd_offset;
    bool is_device_rnd;
    seed_inc_func(increment, &rnd_seed, &offset_ptr, &rnd_offset, &is_device_rnd);
    if (is_device_rnd) {
        *state = at::PhiloxCudaState(rnd_seed, const_cast<int64_t *>(offset_ptr), static_cast<uint32_t>(rnd_offset));
    } else {
        *state = at::PhiloxCudaState(rnd_seed, rnd_offset);
    }
}

static cudaDeviceProp g_prop;

static cudaDeviceProp *GetCurrentDeviceProperties() {
    static std::once_flag flag;   
    std::call_once(flag, [] {
      int dev_id;
      ASSERT_CHECK(cudaGetDevice(&dev_id) == cudaSuccess);
      ASSERT_CHECK(cudaGetDeviceProperties(&g_prop, dev_id) == cudaSuccess);
    });
    return &g_prop;
}   

static void SetZero(void *ptr, size_t sizeof_type, std::initializer_list<int> shapes, cudaStream_t stream) {
    size_t n = sizeof_type;
    for (int s : shapes) n *= s;
    ASSERT_CHECK(cudaMemsetAsync(ptr, 0, n, stream) == cudaSuccess); 
}

int seq_len_round(int real_seq_len, bool use_256_kernel = false) {
 if (!use_256_kernel) {
   if (real_seq_len > 384) {
     return 512;
   } else if (real_seq_len > 128) {
     return 384; 
   } else if (real_seq_len > 0) {
     return 128; 
   } else {
     ASSERT_CHECK(false);
   }
 } else {
   if (real_seq_len > 384) {
     return 512;
   } else if (real_seq_len > 256) {
     return 384; 
   } else if (real_seq_len > 128) {
     return 256; 
   } else if (real_seq_len > 0){
     return 128; 
   } else {
     ASSERT_CHECK(false);
   }
 }
}
  
cudaStream_t stream_384;
cudaStream_t stream_256;
cudaStream_t stream_128;
const bool use_256_kernel = true;

void fmhalib_fwd(const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const void *host_cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout,
                 const int max_seq_len,
                 const bool is_training,
                 const bool is_nl,
                 const bool zero_tensors,
		 SeedIncFuncPtr seed_inc_func,
                 cudaStream_t stream,
                 void *ctx_ptr, // {total, num_heads, head_size}
                 void *s_ptr) { // {batch_size, num_heads, seq_len, seq_len}
    FMHALIB_BEGIN_FUNC
    auto dprops = GetCurrentDeviceProperties(); 
    ASSERT_CHECK(dprops->major == 8 && dprops->minor == 0);
    
    /*
     *cu_seqlens_ptr: [0, 3, 6, 8, 9]
     *seq_len_per_sample: [3, 3, 2, 1]
     *seq_len_group_idx: [0, 6, 8, 9]
     *group_len:[6, 2, 1]
     * */
    const int group_size = use_256_kernel ? 4 : 3;
    std::vector<int> seq_len_per_sample(batch_size); 
    std::vector<int> seq_len_group_idx(group_size);
    std::vector<int> group_len(group_size);
    int cur_group = 0;
    int cur_group_len = 1;
    int cur_idx = 0;
    
    if (is_training) {
      for (int i = 0; i < group_size; i++) {
        seq_len_group_idx[i] = static_cast<const int*>(host_cu_seqlens_ptr)[batch_size];
        group_len[i] = 0; 
      }
      seq_len_group_idx[cur_idx++] = 0;

      for (int i = 0; i < batch_size; i++) {
        seq_len_per_sample[i] = static_cast<const int*>(host_cu_seqlens_ptr)[i + 1]
		- static_cast<const int*>(host_cu_seqlens_ptr)[i];
        // round so as the elements are among [512, 384, 256, 128].
        seq_len_per_sample[i] = seq_len_round(seq_len_per_sample[i], use_256_kernel);
        if (i > 0) {
          if (seq_len_per_sample[i] != seq_len_per_sample[i - 1]) {
	    seq_len_group_idx[cur_idx++] = static_cast<const int*>(host_cu_seqlens_ptr)[i];
            group_len[cur_group++] = cur_group_len;
            cur_group_len = 1;	
	  } else {
            cur_group_len += 1; 
	  } 
        }
      }
      seq_len_group_idx[cur_idx] = static_cast<const int*>(host_cu_seqlens_ptr)[batch_size];
      group_len[cur_group] = cur_group_len;
    }

    if (is_training) {
      int all_group_size = use_256_kernel ?
	      (group_len[0] + group_len[1] + group_len[2] + group_len[3]) :
	      (group_len[0] + group_len[1] + group_len[2]);
      if (all_group_size != batch_size) {
        ASSERT_CHECK(false);
      }
    }

    if (stream_384 == NULL) {
      cudaStreamCreate(&stream_384);
    }
    if (stream_256 == NULL) {
      cudaStreamCreate(&stream_256);
    }
    if (stream_128 == NULL) {
      cudaStreamCreate(&stream_128);
    }

    cudaEvent_t event;
    cudaEvent_t event_384;
    cudaEvent_t event_256;
    cudaEvent_t event_128;
    cudaEvent_t event_512_before;

    cudaEventCreate(&event);
    cudaEventCreate(&event_384);
    cudaEventCreate(&event_256);
    cudaEventCreate(&event_128);
    cudaEventCreate(&event_512_before);

    Launch_params<Fused_multihead_attention_fprop_params> launch_params_512(dprops, 
		    stream, is_training, is_nl);
    Launch_params<Fused_multihead_attention_fprop_params> launch_params_384(dprops, 
		    stream_384, is_training, is_nl);
    Launch_params<Fused_multihead_attention_fprop_params> launch_params_256(dprops, 
		    stream_256, is_training, is_nl);
    Launch_params<Fused_multihead_attention_fprop_params> launch_params_128(dprops, 
		    stream_128, is_training, is_nl);
    
    int seq_len_512 = 512;
    int seq_len_384 = 384;
    int seq_len_256 = 256;
    int seq_len_128 = 128;
    auto launch_512 = &run_fmha_fp16_512_64_sm80;
    auto launch_384 = &run_fmha_fp16_384_64_sm80;
    auto launch_256 = &run_fmha_fp16_256_64_sm80;
    auto launch_128 = &run_fmha_fp16_128_64_sm80;

    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    if( zero_tensors ) {
        SetZero(ctx_ptr, 2, {total, num_heads, head_size}, stream);  
        SetZero(s_ptr, 2, {batch_size, num_heads, 512, 512}, stream);
    }
    cudaEventRecord(event_512_before, stream);
    
    set_params(launch_params_512.params,
	       // Note: only training use multiple fmha kernel methods!
               is_training ? group_len[0] : batch_size, 
               seq_len_512,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               ctx_ptr,
               s_ptr,
               p_dropout);

    if (is_training && group_len[1] > 0) {
      int qkv_offset = seq_len_group_idx[1] * head_size * num_heads * 3;
      const __half* new_qkv_ptr = static_cast<const __half*>(qkv_ptr) + qkv_offset;
      const int* new_cu_seqlens_ptr = static_cast<const int*>(cu_seqlens_ptr) + group_len[0];
      // {total, num_heads, head_size}
      __half* new_ctx_ptr = static_cast<__half*>(ctx_ptr) 
	      + seq_len_group_idx[1] * num_heads * head_size;
      // batch_size, num_heads, seq_len, seq_len
      __half* new_s_ptr = static_cast<__half*>(s_ptr) 
	      + group_len[0] * num_heads * 512 * 512; 
      set_params(launch_params_384.params,
               group_len[1],  
               seq_len_384,
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr)),
               static_cast<void*>(new_ctx_ptr),
               static_cast<void*>(new_s_ptr),
               p_dropout);
    }
    if (!use_256_kernel) {
      // use 128 kernels.
      if (is_training  && group_len[2] > 0) {
        int qkv_offset = seq_len_group_idx[2] * head_size * num_heads * 3;
        const __half* new_qkv_ptr = static_cast<const __half*>(qkv_ptr) + qkv_offset;
        const int* new_cu_seqlens_ptr = static_cast<const int*>(cu_seqlens_ptr)
		+ group_len[0] + group_len[1];
        __half* new_ctx_ptr = static_cast<__half*>(ctx_ptr)
		+ seq_len_group_idx[2] * num_heads * head_size;
        __half* new_s_ptr = static_cast<__half*>(s_ptr)
		+ (group_len[0] + group_len[1]) * num_heads * 512 * 512;
        set_params(launch_params_128.params,
               group_len[2], 
               seq_len_128,
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr)),
               static_cast<void*>(new_ctx_ptr),
               static_cast<void*>(new_s_ptr),
               p_dropout);
      }
    } else {
      // use 256_kernel
      if (is_training  && group_len[2] > 0) {
        int qkv_offset = seq_len_group_idx[2] * head_size * num_heads * 3;
        const __half* new_qkv_ptr = static_cast<const __half*>(qkv_ptr) + qkv_offset;
        const int* new_cu_seqlens_ptr = static_cast<const int*>(cu_seqlens_ptr)
		+ group_len[0] + group_len[1];
        __half* new_ctx_ptr = static_cast<__half*>(ctx_ptr) 
		+ seq_len_group_idx[2] * num_heads * head_size;
        __half* new_s_ptr = static_cast<__half*>(s_ptr)
		+ (group_len[0] + group_len[1]) * num_heads * 512 * 512;
        set_params(launch_params_256.params,
               group_len[2], 
               seq_len_256,
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr)),
               static_cast<void*>(new_ctx_ptr),
               static_cast<void*>(new_s_ptr),
               p_dropout);
      }
      if (is_training  && group_len[3] > 0) {
        int qkv_offset = seq_len_group_idx[3] * head_size * num_heads * 3;
        const __half* new_qkv_ptr = static_cast<const __half*>(qkv_ptr) + qkv_offset;
        const int* new_cu_seqlens_ptr = static_cast<const int*>(cu_seqlens_ptr) 
		+ group_len[0] + group_len[1] + group_len[2]; 
        __half* new_ctx_ptr = static_cast<__half*>(ctx_ptr) 
		+ seq_len_group_idx[3] * num_heads * head_size;
        __half* new_s_ptr = static_cast<__half*>(s_ptr) 
		+ (group_len[0] + group_len[1] + group_len[2]) * num_heads * 512 * 512; 
        set_params(launch_params_128.params,
               group_len[3], // batch_size,
               seq_len_128,
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr)),
               static_cast<void*>(new_ctx_ptr),
               static_cast<void*>(new_s_ptr),
               p_dropout);
      }
    }

    launch_512(launch_params_512, /*configure=*/ true);
    
    if (is_training && group_len[1] > 0) {
      cudaStreamWaitEvent(stream_384, event_512_before);
      launch_384(launch_params_384, /*configure=*/ true);
    }
    if (!use_256_kernel) {
      if (is_training && group_len[2] > 0) {
        cudaStreamWaitEvent(stream_128, event_512_before);
        launch_128(launch_params_128, /*configure=*/ true);
      }
    } else {
      if (is_training && group_len[2] > 0) {
        cudaStreamWaitEvent(stream_256, event_512_before);
        launch_256(launch_params_256, /*configure=*/ true);
      }
      if (is_training && group_len[3] > 0) {
        cudaStreamWaitEvent(stream_128, event_512_before);
        launch_128(launch_params_128, /*configure=*/ true);
      }
    }

    if ( is_training ) {
        int64_t counter_offset = launch_params_512.elts_per_thread;
        SetPhiloxCudaState(&launch_params_512.params.philox_args, seed_inc_func, counter_offset);
	if (group_len[1] > 0) {
          int64_t counter_offset = launch_params_384.elts_per_thread;
          SetPhiloxCudaState(&launch_params_384.params.philox_args, seed_inc_func, counter_offset);
	}
        if (!use_256_kernel) {	
	  if (group_len[2] > 0) {
            int64_t counter_offset = launch_params_128.elts_per_thread;
            SetPhiloxCudaState(&launch_params_128.params.philox_args, seed_inc_func, counter_offset);
	  }
	} else {
	  if (group_len[2] > 0) {
            int64_t counter_offset = launch_params_256.elts_per_thread;
            SetPhiloxCudaState(&launch_params_256.params.philox_args, seed_inc_func, counter_offset);
	  }
	  if (group_len[3] > 0) {
            int64_t counter_offset = launch_params_128.elts_per_thread;
            SetPhiloxCudaState(&launch_params_128.params.philox_args, seed_inc_func, counter_offset);
	  }
	} 	
    }
    
    launch_512(launch_params_512, /*configure=*/ false);
    cudaEventRecord(event, stream);

    if (is_training && group_len[1] > 0) {
      launch_384(launch_params_384, /*configure=*/ false);
      cudaEventRecord(event_384, stream_384);
    }
    if (!use_256_kernel) {
      if (is_training && group_len[2] > 0) {
        launch_128(launch_params_128, /*configure=*/ false);
        cudaEventRecord(event_128, stream_128);
      }
    } else {
      if (is_training && group_len[2] > 0) {
        launch_256(launch_params_256, /*configure=*/ false);
        cudaEventRecord(event_256, stream_256);
      }
      if (is_training && group_len[3] > 0) {
        launch_128(launch_params_128, /*configure=*/ false);
        cudaEventRecord(event_128, stream_128);
      }
    }
    
    // stream will go on executing until events on stream_384/256/128 finishes.
    if (is_training) {
      cudaStreamWaitEvent(stream, event);
      cudaStreamWaitEvent(stream, event_384);
      cudaStreamWaitEvent(stream, event_128);
      if (use_256_kernel) {
        cudaStreamWaitEvent(stream, event_256);
      }
    }

    cudaEventDestroy(event);
    cudaEventDestroy(event_384);
    cudaEventDestroy(event_128);
    cudaEventDestroy(event_256);

    FMHALIB_END_FUNC
}

static void fmhalib_bwd_nl(const void *dout_ptr,
                           const void *qkv_ptr,
                           const void *cu_seqlens_ptr,
                           const int total,
                           const int num_heads,
                           const int head_size,
                           const int batch_size,
                           const float p_dropout,         // probability to drop
                           const int max_seq_len,         // max sequence length to choose the kernel
                           const bool zero_tensors,
                           cudaStream_t stream,
                           void *softmax_ptr,             // will be overwritten
                           void *dqkv_ptr,
                           void *dkv_ptr) {
    FMHALIB_BEGIN_FUNC
    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    int seq_len = 512;
    auto launch = &run_fmha_dgrad_fp16_512_64_sm80_nl;

    if( zero_tensors ) {
        SetZero(dqkv_ptr, 2, {total, num_heads, 3, head_size}, stream); 
    }

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               const_cast<void*>(dout_ptr),     // o_ptr = dout
               softmax_ptr,  // softmax gets overwritten by dP!
               p_dropout);

    params.dkv_ptr = dkv_ptr;

    Data_type acc_type = DATA_TYPE_FP32;
    set_alpha(params.scale_bmm1, 1.f, acc_type);
    set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
    params.dqkv_ptr = dqkv_ptr;

    auto num_chunks = fmhalib_bwd_nl_num_chunks(batch_size);
    launch(params, num_chunks, stream);

    //SPLIT-K reduction of num_chunks dK, dV parts

    // The equivalent of the following Pytorch code:
    // using namespace torch::indexing;
    // at::Tensor view_out = dqkv.index({Slice(), Slice(1, None, None)});
    // torch::sum_out(view_out, dkv, 1);

    const int hidden_size = num_heads * head_size;
    fmha_run_noloop_reduce(
        dqkv_ptr, dkv_ptr, static_cast<const int *>(cu_seqlens_ptr), hidden_size, batch_size, total, num_chunks, stream);
    FMHALIB_END_FUNC
}


void fmhalib_bwd(const void *dout_ptr,
                 const void *qkv_ptr,
                 const void *cu_seqlens_ptr,
                 const void *host_cu_seqlens_ptr,
                 const int total,
                 const int num_heads,
                 const int head_size,
                 const int batch_size,
                 const float p_dropout,
                 const int max_seq_len,
                 const bool is_nl,
                 const bool zero_tensors,
                 cudaStream_t stream,
                 void *softmax_ptr,  // will be overwritten
                 void *dqkv_ptr,
                 void *workspace,
                 uint64_t *workspace_size) {
    if (dqkv_ptr == nullptr) {
        if (is_nl) {
            int num_chunks = fmhalib_bwd_nl_num_chunks(batch_size); 
            *workspace_size = static_cast<uint64_t>(total) * num_chunks * 2 * num_heads * head_size * 2;  
        } else {
            *workspace_size = 0;
        }
        return;
    }

    if (is_nl) {
        fmhalib_bwd_nl(dout_ptr, qkv_ptr, cu_seqlens_ptr, total, num_heads, head_size,
            batch_size, p_dropout, max_seq_len, zero_tensors, stream, softmax_ptr,
            dqkv_ptr, workspace);
        return;
    }

    FMHALIB_BEGIN_FUNC
    auto dprops = GetCurrentDeviceProperties();
    ASSERT_CHECK(dprops->major == 8 && dprops->minor == 0);
    
    const int group_size = use_256_kernel ? 4 : 3;
    std::vector<int> seq_len_per_sample(batch_size); 
    std::vector<int> seq_len_group_idx(group_size);
    std::vector<int> group_len(group_size);
    
    for (int i = 0; i < group_size; i++) {
      seq_len_group_idx[i] = static_cast<const int*>(host_cu_seqlens_ptr)[batch_size];
      group_len[i] = 0; 
    }

    int cur_group = 0;
    int cur_group_len = 1;
    int cur_idx = 0;
    seq_len_group_idx[cur_idx++] = 0; 
    for (int i = 0; i < batch_size; i++) {
      seq_len_per_sample[i] = static_cast<const int*>(host_cu_seqlens_ptr)[i + 1] 
	      - static_cast<const int*>(host_cu_seqlens_ptr)[i];
      // round so as the elements in array is among [512, 384, 256, 128].
      seq_len_per_sample[i] = seq_len_round(seq_len_per_sample[i], use_256_kernel);
      if (i > 0) {
        if (seq_len_per_sample[i] != seq_len_per_sample[i - 1]) {
	  seq_len_group_idx[cur_idx++] = static_cast<const int*>(host_cu_seqlens_ptr)[i];
          group_len[cur_group++] = cur_group_len;
          cur_group_len = 1;	
	} else {
          cur_group_len += 1; 
	} 
      }
    }
    seq_len_group_idx[cur_idx] = static_cast<const int*>(host_cu_seqlens_ptr)[batch_size];
    group_len[cur_group] = cur_group_len;
    
    int all_group_size = use_256_kernel ?
	    (group_len[0] + group_len[1] + group_len[2] + group_len[3]):
	    (group_len[0] + group_len[1] + group_len[2]);
    if (all_group_size != batch_size) {
        ASSERT_CHECK(false);
    }

    int seq_len_512 = 512;
    int seq_len_384 = 384;
    int seq_len_256 = 256;
    int seq_len_128 = 128;
    auto launch_512 = &run_fmha_dgrad_fp16_512_64_sm80;
    auto launch_384 = &run_fmha_dgrad_fp16_384_64_sm80;
    auto launch_256 = &run_fmha_dgrad_fp16_256_64_sm80;
    auto launch_128 = &run_fmha_dgrad_fp16_128_64_sm80;

    if (stream_384 == NULL) {
      cudaStreamCreate(&stream_384);
    }
    if (stream_256 == NULL) {
      cudaStreamCreate(&stream_256);
    }
    if (stream_128 == NULL) {
      cudaStreamCreate(&stream_128);
    }

    cudaEvent_t event;
    cudaEvent_t event_384;
    cudaEvent_t event_256;
    cudaEvent_t event_128;
    cudaEvent_t event_512_before;

    cudaEventCreate(&event);
    cudaEventCreate(&event_384);
    cudaEventCreate(&event_256);
    cudaEventCreate(&event_128);
    cudaEventCreate(&event_512_before);
    
    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    if( zero_tensors ) {
        SetZero(dqkv_ptr, 2, {total, num_heads, 3, head_size}, stream);
    }
    // record the op precedding fmha. 
    cudaEventRecord(event_512_before, stream);

    Fused_multihead_attention_fprop_params params;
    Fused_multihead_attention_fprop_params params_384;
    Fused_multihead_attention_fprop_params params_256;
    Fused_multihead_attention_fprop_params params_128;

    set_params(params,
               group_len[0], 
               seq_len_512, 
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               const_cast<void*>(dout_ptr),     // we set o_ptr to dout
               softmax_ptr,  // softmax gets overwritten by dP!
               p_dropout);
    params.dqkv_ptr = dqkv_ptr;

    int qkv_offset = seq_len_group_idx[1] * head_size * num_heads * 3;
    int output_offset = seq_len_group_idx[1] * head_size * num_heads;
    const __half* new_qkv_ptr = static_cast<const __half*>(qkv_ptr) + qkv_offset;
    const int* new_cu_seqlens_ptr = static_cast<const int*>(cu_seqlens_ptr) + group_len[0];
    const __half* new_dout_ptr = static_cast<const __half*>(dout_ptr) + output_offset;
    __half* new_softmax_ptr = static_cast<__half*>(softmax_ptr)
	    + group_len[0] * num_heads * 512 * 512;
    if (group_len[1] > 0) {
      set_params(params_384,
               group_len[1],  
               seq_len_384,  
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr)),
               const_cast<void*>(static_cast<const void*>(new_dout_ptr)),     // we set o_ptr to dout
               static_cast<void*>(new_softmax_ptr),  // softmax gets overwritten by dP!
               p_dropout);
    }
    params_384.dqkv_ptr = static_cast<void*>(static_cast<__half*>(dqkv_ptr) + qkv_offset);
   
    if (!use_256_kernel) {
      int qkv_offset_2 = seq_len_group_idx[2] * head_size * num_heads * 3;
      int output_offset_2 = seq_len_group_idx[2] * head_size * num_heads;
      const __half* new_qkv_ptr_2 = static_cast<const __half*>(qkv_ptr) + qkv_offset_2;
      const int* new_cu_seqlens_ptr_2 = static_cast<const int*>(cu_seqlens_ptr)
	      + group_len[0] + group_len[1];
      const __half* new_dout_ptr_2 = static_cast<const __half*>(dout_ptr) + output_offset_2;
      // {batch_size, num_heads, seq_len, seq_len}
      __half* new_softmax_ptr_2 = static_cast<__half*>(softmax_ptr)
	      + (group_len[0] + group_len[1]) * num_heads * 512 * 512;
      if (group_len[2] > 0) {
        set_params(params_128,
               group_len[2], 
               seq_len_128,  
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr_2)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr_2)),
               const_cast<void*>(static_cast<const void*>(new_dout_ptr_2)),     // we set o_ptr to dout
               static_cast<void*>(new_softmax_ptr_2),  // softmax gets overwritten by dP!
               p_dropout);
      }
      params_128.dqkv_ptr = static_cast<void*>(static_cast<__half*>(dqkv_ptr) + qkv_offset_2);
    } else {
      // for 256 kernel.
      int qkv_offset_2 = seq_len_group_idx[2] * head_size * num_heads * 3;
      int output_offset_2 = seq_len_group_idx[2] * head_size * num_heads;
      const __half* new_qkv_ptr_2 = static_cast<const __half*>(qkv_ptr) + qkv_offset_2;
      const int* new_cu_seqlens_ptr_2 = static_cast<const int*>(cu_seqlens_ptr)
	      + group_len[0] + group_len[1];
      const __half* new_dout_ptr_2 = static_cast<const __half*>(dout_ptr) + output_offset_2;
      // {batch_size, num_heads, seq_len, seq_len}
      __half* new_softmax_ptr_2 = static_cast<__half*>(softmax_ptr)
	      + (group_len[0] + group_len[1]) * num_heads * 512 * 512;
      if (group_len[2] > 0) {
        set_params(params_256,
               group_len[2], 
               seq_len_256,  
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr_2)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr_2)),
               const_cast<void*>(static_cast<const void*>(new_dout_ptr_2)),     // we set o_ptr to dout
               static_cast<void*>(new_softmax_ptr_2),  // softmax gets overwritten by dP!
               p_dropout);
      }
      params_256.dqkv_ptr = static_cast<void*>(static_cast<__half*>(dqkv_ptr) + qkv_offset_2);
      
      //  for 128 kernel.
      int qkv_offset_3 = seq_len_group_idx[3] * head_size * num_heads * 3;
      int output_offset_3 = seq_len_group_idx[3] * head_size * num_heads;
      const __half* new_qkv_ptr_3 = static_cast<const __half*>(qkv_ptr) + qkv_offset_3;
      const int* new_cu_seqlens_ptr_3 = static_cast<const int*>(cu_seqlens_ptr) 
	      + group_len[0] + group_len[1] + group_len[2];
      const __half* new_dout_ptr_3 = static_cast<const __half*>(dout_ptr) + output_offset_3;
      // {batch_size, num_heads, seq_len, seq_len}
      __half* new_softmax_ptr_3 = static_cast<__half*>(softmax_ptr) 
	      + (group_len[0] + group_len[1] + group_len[2]) * num_heads * 512 * 512; 
      if (group_len[3] > 0) {
        set_params(params_128,
               group_len[3], 
               seq_len_128,  
               num_heads,
               head_size,
               const_cast<void*>(static_cast<const void*>(new_qkv_ptr_3)),
               const_cast<void*>(static_cast<const void*>(new_cu_seqlens_ptr_3)),
               const_cast<void*>(static_cast<const void*>(new_dout_ptr_3)),     // we set o_ptr to dout
               static_cast<void*>(new_softmax_ptr_3),  // softmax gets overwritten by dP!
               p_dropout);
      }
      params_128.dqkv_ptr = static_cast<void*>(static_cast<__half*>(dqkv_ptr) + qkv_offset_3);
    }

    // we're re-using these scales
    Data_type acc_type = DATA_TYPE_FP32;
    set_alpha(params.scale_bmm1, 1.f, acc_type);
    set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
    
    set_alpha(params_384.scale_bmm1, 1.f, acc_type);
    set_alpha(params_384.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params_384.scale_bmm2, 1.f, DATA_TYPE_FP16);
    
    set_alpha(params_128.scale_bmm1, 1.f, acc_type);
    set_alpha(params_128.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params_128.scale_bmm2, 1.f, DATA_TYPE_FP16);

    if (use_256_kernel) {
      set_alpha(params_256.scale_bmm1, 1.f, acc_type);
      set_alpha(params_256.scale_softmax, 1.f / sqrtf(head_size), acc_type);
      set_alpha(params_256.scale_bmm2, 1.f, DATA_TYPE_FP16);
    } 

    launch_512(params, stream);
    cudaEventRecord(event, stream);
      
    if (group_len[1] > 0) {
      // begin to exec after the precedding op of fmha finishes.
      cudaStreamWaitEvent(stream_384, event_512_before);
      launch_384(params_384, stream_384);
      cudaEventRecord(event_384, stream_384);
    }
    if (!use_256_kernel) {
      if (group_len[2] > 0) {
        // begin to exec after the precedding op of fmha finishes.
        cudaStreamWaitEvent(stream_128, event_512_before);
        launch_128(params_128, stream_128);
        cudaEventRecord(event_128, stream_128);
      }
    } else {
      if (group_len[2] > 0) {
        cudaStreamWaitEvent(stream_256, event_512_before);
        launch_256(params_256, stream_256);
        cudaEventRecord(event_256, stream_256);
      }
      if (group_len[3] > 0) {
        cudaStreamWaitEvent(stream_128, event_512_before);
        launch_128(params_128, stream_128);
        cudaEventRecord(event_128, stream_128);
      }
    }
    cudaStreamWaitEvent(stream, event);
    cudaStreamWaitEvent(stream, event_384);
    cudaStreamWaitEvent(stream, event_128);
    if (use_256_kernel) {
      cudaStreamWaitEvent(stream, event_256);
    }

    cudaEventDestroy(event);
    cudaEventDestroy(event_384);
    cudaEventDestroy(event_256);
    cudaEventDestroy(event_128);

    FMHALIB_END_FUNC
}

#ifdef __cplusplus
}
#endif
