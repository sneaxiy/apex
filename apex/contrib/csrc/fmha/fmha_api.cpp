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
#include <memory>
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
#include "fmhalib.h"

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

void fmhalib_clear_error() {
  fmhalib_err_msg = nullptr;
}

void fmhalib_set_error(const char *msg) {
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

#define FMHALIB_BEGIN_FUNC fmhalib_clear_error(); try {
#define FMHALIB_END_FUNC } catch (::std::exception &__e) { fmhalib_set_error(__e.what()); } catch (...) { fmhalib_set_error(nullptr); } 

void set_params(Fused_multihead_attention_fprop_params &params,
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

    set_alpha(params.scale_bmm1, scale_bmm1, acc_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    params.rp_dropout = 1.f / params.p_dropout;
    ASSERT_CHECK(p_dropout < 1.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);
}

int fmhalib_random_increment(const int seq_len) {
    constexpr int warps_m = 1;
    constexpr int warps_n = 4;  // this leads to an upper bound
    const int mmas_m = seq_len / 16 / warps_m;
    const int mmas_n = seq_len / 16 / warps_n;

    const int elts_per_thread = 8 * mmas_m * mmas_n;
    return elts_per_thread;
}

int fmhalib_bwd_nl_num_chunks(const int batch_size) {
    int num_chunks = 2;
    if( batch_size == 1 ) {
        num_chunks = 4;
    }else if( batch_size == 2 ) {
        num_chunks = 3;
    }
    return num_chunks;
}

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
	         void *s_ptr) {
    FMHALIB_BEGIN_FUNC
    int seq_len;
    auto launch = &run_fmha_fp16_512_64_sm80;
    if (max_seq_len == 128) {
        seq_len = 128;
        launch = &run_fmha_fp16_128_64_sm80;
    } else if (max_seq_len == 256) {
        seq_len = 256;
        launch = &run_fmha_fp16_256_64_sm80;
    } else if (max_seq_len == 384) {
        seq_len = 384;
        launch = &run_fmha_fp16_384_64_sm80;
    } else if (max_seq_len == 512) {
        seq_len = 512;
        launch = &run_fmha_fp16_512_64_sm80;
    } else {
        ASSERT_CHECK(false);
    }

    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    Fused_multihead_attention_fprop_params params;
    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               ctx_ptr,
               s_ptr,
               p_dropout);

    if (is_training) {
	if (is_device_rnd) {
	    params.philox_args = at::PhiloxCudaState(rnd_seed, const_cast<int64_t *>(offset_ptr), static_cast<uint32_t>(rnd_offset));
        } else {
	    params.philox_args = at::PhiloxCudaState(rnd_seed, rnd_offset);
	}
    }

    launch(params, is_training, stream);
    FMHALIB_END_FUNC
}

void fmhalib_bwd(const void *dout_ptr,
	         const void *qkv_ptr,
	         const void *cu_seqlens_ptr,
	         const int total,
                 const int num_heads,
                 const int head_size,
	         const int batch_size,
                 const float p_dropout,         // probability to drop
                 const int max_seq_len,         // max sequence length to choose the kernel
		 cudaStream_t stream,
		 void *softmax_ptr,             // will be overwritten 
		 void *dqkv_ptr 
) {
    FMHALIB_BEGIN_FUNC
    int seq_len = 512;
    auto launch = &run_fmha_dgrad_fp16_512_64_sm80;
    if( max_seq_len == 128 ) {
        seq_len = 128;
        launch = &run_fmha_dgrad_fp16_128_64_sm80;
    } else if( max_seq_len == 256 ) {
        seq_len = 256;
        launch = &run_fmha_dgrad_fp16_256_64_sm80;
    } else if( max_seq_len == 384 ) {
        seq_len = 384;
        launch = &run_fmha_dgrad_fp16_384_64_sm80;
    } else if( max_seq_len == 512 ) {
        seq_len = 512;
        launch = &run_fmha_dgrad_fp16_512_64_sm80;
    } else {
	ASSERT_CHECK(false);
    }

    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               const_cast<void*>(dout_ptr), // we set o_ptr to dout
               softmax_ptr,  // softmax gets overwritten by dP!
               p_dropout);

    // we're re-using these scales
    Data_type acc_type = DATA_TYPE_FP32;
    set_alpha(params.scale_bmm1, 1.f, acc_type);
    set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
    params.dqkv_ptr = dqkv_ptr;

    launch(params, stream);
    FMHALIB_END_FUNC
}

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
                    void *s_ptr) {
    FMHALIB_BEGIN_FUNC
    int seq_len = 512;
    auto launch = &run_fmha_fp16_512_64_sm80_nl;
    ASSERT_CHECK(max_seq_len == seq_len);

    constexpr int warps_m = 1;
    constexpr int warps_n = 4;  // this leads to an upper bound
    const int mmas_m = seq_len / 16 / warps_m;
    const int mmas_n = seq_len / 16 / warps_n;
    // static_assert( mmas_m == 32 );
    // static_assert( mmas_n == 4 );
    const int elts_per_thread = 8 * mmas_m * mmas_n;

    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    Fused_multihead_attention_fprop_params params;

    set_params(params,
               batch_size,
               seq_len,
               num_heads,
               head_size,
               const_cast<void*>(qkv_ptr),
               const_cast<void*>(cu_seqlens_ptr),
               ctx_ptr,
               s_ptr,
               p_dropout);

    // number of times random will be generated per thread, to offset philox counter in the random
    // state

    if( is_training ) {
        if (is_device_rnd) {
            params.philox_args = at::PhiloxCudaState(rnd_seed, const_cast<int64_t *>(offset_ptr), static_cast<uint32_t>(rnd_offset));
        } else {
            params.philox_args = at::PhiloxCudaState(rnd_seed, rnd_offset);
        }
    }
    int num_chunks = 3;
    if(batch_size == 3) {
        num_chunks = 2;
    }

    launch(params, is_training, num_chunks, stream);
    FMHALIB_END_FUNC
}

void fmhalib_bwd_nl(const void *dout_ptr,
                    const void *qkv_ptr,
                    const void *cu_seqlens_ptr,
                    const int total,
                    const int num_heads,
                    const int head_size,
                    const int batch_size,
                    const float p_dropout,         // probability to drop
                    const int max_seq_len,         // max sequence length to choose the kernel
                    cudaStream_t stream,
                    void *softmax_ptr,             // will be overwritten
                    void *dqkv_ptr,
		    void *dkv_ptr
) {
    FMHALIB_BEGIN_FUNC

    ASSERT_CHECK(batch_size > 0);
    ASSERT_CHECK(head_size == 64);

    int seq_len = 512;
    ASSERT_CHECK(max_seq_len == seq_len);
    auto launch = &run_fmha_dgrad_fp16_512_64_sm80_nl;

    int num_chunks = fmhalib_bwd_nl_num_chunks(batch_size);

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

    launch(params, num_chunks, stream);

    //SPLIT-K reduction of num_chunks dK, dV parts

    // The equivalent of the following Pytorch code:
    // using namespace torch::indexing;
    // at::Tensor view_out = dqkv.index({Slice(), Slice(1, None, None)});
    // torch::sum_out(view_out, dkv, 1);

    const int hidden_size = num_heads * head_size;
    fmha_run_noloop_reduce(
        dqkv_ptr, dkv_ptr, reinterpret_cast<const int*>(cu_seqlens_ptr), hidden_size, batch_size, total, num_chunks, stream);

    FMHALIB_END_FUNC
}

#ifdef __cplusplus
}
#endif
