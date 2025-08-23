#include "../include/utils.hpp"

void MemCpy_Tensor(Tensor *to, const Tensor *from, long long to_offset,
                    long long from_offset, size_t num_elem,
                    bool copy_host, bool copy_device, hipStream_t stream) {
    if (copy_host) {
        memcpy(to->buf + 1ll * to_offset, from->buf + 1ll * from_offset, num_elem * sizeof(float));
    }
    
    if (copy_device) {
        if (to->dtype == DType::BF16) {
            bf16 *to_d_buf_bf16 = (bf16 *)to->d_buf + 1ll * to_offset;
            bf16 *from_d_buf_bf16 = (bf16 *)from->d_buf + 1ll * from_offset;
        
            CHECK_HIP(hipMemcpyAsync(to_d_buf_bf16, from_d_buf_bf16, num_elem * sizeof(bf16), hipMemcpyDeviceToDevice, stream));
        } else {
            float *to_d_buf_fp32 = (float *)to->d_buf + 1ll * to_offset;
            float *from_d_buf_fp32 = (float *)from->d_buf + 1ll * from_offset;
        
            CHECK_HIP(hipMemcpyAsync(to_d_buf_fp32, from_d_buf_fp32, num_elem * sizeof(float), hipMemcpyDeviceToDevice, stream));
        }
        
        // Haven't implemented different types for Tensor device
    }
}

void MemSet_Tensor(Tensor *in, int value, bool set_host,
                    bool set_device, hipStream_t stream) {
    size_t num_elem = in->num_elem();
    if (set_host) {
        memset(in->buf, 0, in->num_elem() * sizeof(float));
    }

    else {
        CHECK_HIP(hipMemsetAsync(in->d_buf, 0, in->num_elem() * in->get_dtype_size(), stream));
    }
}
