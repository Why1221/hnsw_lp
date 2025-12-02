#pragma once

#include "hnswlib.h"
#include <cmath>
#include <stdexcept>
#include "pow.h" // Your provided pow.h header from nmslib

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

namespace hnswlib {

// Forward declarations of all distance functions
static float LpDistance_p05_scalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static float LpDistance_p15_scalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static float LpDistance_p05_sse_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static float LpDistance_p05_avx_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static float LpDistance_p05_avx512_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static DISTFUNC<float> L05SqrSIMD16Ext;
static float LpDistance_p05_16_residuals(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

static float LpDistance_p15_sse_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static float LpDistance_p15_avx_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static float LpDistance_p15_avx512_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
static DISTFUNC<float> L15SqrSIMD16Ext;
static float LpDistance_p15_16_residuals(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
#endif


class LpSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    char* dist_func_param_proxy_;
public:
    PowerProxyObject<float> pow_p_;
    float p_;

    static float LpDistanceWrapper(const void* pVect1, const void* pVect2, const void* qty_ptr) {
        const char* param_buffer = static_cast<const char*>(qty_ptr);
        size_t qty = *((const size_t*)param_buffer);
        LpSpace* instance = *((LpSpace**)(param_buffer + sizeof(size_t)));
        
        const float* vect1 = static_cast<const float*>(pVect1);
        const float* vect2 = static_cast<const float*>(pVect2);
        
        float total_sum = 0.0f;
        for (size_t i = 0; i < qty; i++) {
            total_sum += instance->pow_p_.pow(std::abs(vect1[i] - vect2[i]));
        }
        
        return total_sum;
    }

    explicit LpSpace(size_t dim, float p) : dim_(dim), dist_func_param_proxy_(nullptr), pow_p_(p, 18), p_(p) {
        if (p <= 0) {
            throw std::invalid_argument("The p value must be positive.");
        }

        if (p == 0.5f) {
            #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                L05SqrSIMD16Ext = LpDistance_p05_sse_16; // Default to SSE
                #if defined(USE_AVX512)
                    if (AVX512Capable()) L05SqrSIMD16Ext = LpDistance_p05_avx512_16;
                    else if (AVXCapable()) L05SqrSIMD16Ext = LpDistance_p05_avx_16;
                #elif defined(USE_AVX)
                    if (AVXCapable()) L05SqrSIMD16Ext = LpDistance_p05_avx_16;
                #endif
                fstdistfunc_ = LpDistance_p05_16_residuals;
                std::cout << "Using dedicated multi-level SIMD implementation for p=0.5" << std::endl;
            #else
                fstdistfunc_ = LpDistance_p05_scalar;
                std::cout << "Using scalar implementation for p=0.5" << std::endl;
            #endif
        } 
        else if (p == 1.5f) {
            #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                L15SqrSIMD16Ext = LpDistance_p15_sse_16; // Default to SSE
                #if defined(USE_AVX512)
                    if (AVX512Capable()) L15SqrSIMD16Ext = LpDistance_p15_avx512_16;
                    else if (AVXCapable()) L15SqrSIMD16Ext = LpDistance_p15_avx_16;
                #elif defined(USE_AVX)
                    if (AVXCapable()) L15SqrSIMD16Ext = LpDistance_p15_avx_16;
                #endif
                fstdistfunc_ = LpDistance_p15_16_residuals;
                std::cout << "Using dedicated multi-level SIMD implementation for p=1.5" << std::endl;
            #else
                fstdistfunc_ = LpDistance_p15_scalar;
                std::cout << "Using scalar implementation for p=1.5" << std::endl;
            #endif
        }
        else { 
            if (p == 1.0f || p == 2.0f) {
                std::cout << "Warning: LpSpace is being used for p=" << p 
                          << ". For optimal performance, consider HNSWlib's built-in L1Space or L2Space." << std::endl;
            }
            std::cout << "Using PowerProxyObject (scalar) implementation for p = " << p << std::endl;
            fstdistfunc_ = LpSpace::LpDistanceWrapper;
            
            size_t param_size = sizeof(size_t) + sizeof(LpSpace*);
            dist_func_param_proxy_ = new char[param_size];
            *((size_t*)dist_func_param_proxy_) = dim;
            *((LpSpace**)(dist_func_param_proxy_ + sizeof(size_t))) = this;
        }
        
        data_size_ = dim * sizeof(float);
    }

    ~LpSpace() {
        if (dist_func_param_proxy_) delete[] dist_func_param_proxy_;
    }

    size_t get_data_size() { return data_size_; }
    DISTFUNC<float> get_dist_func() { return fstdistfunc_; }
    void* get_dist_func_param() {
        if (fstdistfunc_ == LpSpace::LpDistanceWrapper) return dist_func_param_proxy_;
        return &dim_;
    }
};

//================================================================================
// DEFINITIONS of all distance functions
//================================================================================

// --- L0.5 Implementations ---
float LpDistance_p05_scalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    float total_sum_of_sqrt = 0;
    for (size_t i = 0; i < qty; ++i) {
        total_sum_of_sqrt += std::sqrt(std::abs(pVect1[i] - pVect2[i]));
    }
    return total_sum_of_sqrt * total_sum_of_sqrt;
}

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
float LpDistance_p05_sse_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m128 sum = _mm_setzero_ps();
    __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    while (pVect1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        __m128 v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        __m128 diff = _mm_sub_ps(v1, v2);
        __m128 abs_diff = _mm_and_ps(diff, abs_mask);
        sum = _mm_add_ps(sum, _mm_sqrt_ps(abs_diff));
    }
    
    PORTABLE_ALIGN32 float TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float LpDistance_p05_avx_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m256 sum = _mm256_setzero_ps();
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        __m256 diff = _mm256_sub_ps(v1, v2);
        __m256 abs_diff = _mm256_and_ps(diff, abs_mask);
        sum = _mm256_add_ps(sum, _mm256_sqrt_ps(abs_diff));
    }
    
    PORTABLE_ALIGN32 float TmpRes[8];
    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

float LpDistance_p05_avx512_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m512 sum = _mm512_setzero_ps();
    
    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
        __m512 diff = _mm512_sub_ps(v1, v2);
        __m512 abs_diff = _mm512_abs_ps(diff);
        sum = _mm512_add_ps(sum, _mm512_sqrt_ps(abs_diff));
    }
    
    return _mm512_reduce_add_ps(sum);
}

float LpDistance_p05_16_residuals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty_simd = qty - (qty % 16);
    
    float res_simd_sum_of_sqrt = L05SqrSIMD16Ext(pVect1v, pVect2v, &qty_simd);
    
    const float* pVect1 = static_cast<const float*>(pVect1v) + qty_simd;
    const float* pVect2 = static_cast<const float*>(pVect2v) + qty_simd;
    size_t qty_left = qty - qty_simd;
    float res_scalar_sum_of_sqrt = 0;
    for(size_t i=0; i<qty_left; ++i){
        res_scalar_sum_of_sqrt += std::sqrt(std::abs(pVect1[i] - pVect2[i]));
    }

    float total_sum_of_sqrt = res_simd_sum_of_sqrt + res_scalar_sum_of_sqrt;
    return total_sum_of_sqrt * total_sum_of_sqrt;
}
#endif

// --- L1.5 Implementations ---
float LpDistance_p15_scalar(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    float total_sum = 0;
    for (size_t i = 0; i < qty; ++i) {
        float abs_diff = std::abs(pVect1[i] - pVect2[i]);
        total_sum += abs_diff * std::sqrt(abs_diff);
    }
    return std::pow(total_sum, 1.0f/1.5f);
}

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
float LpDistance_p15_sse_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m128 sum = _mm_setzero_ps();
    __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

    while (pVect1 < pEnd1) {
        __m128 v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        __m128 v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        __m128 diff = _mm_sub_ps(v1, v2);
        __m128 abs_diff = _mm_and_ps(diff, abs_mask);
        __m128 sqrt_val = _mm_sqrt_ps(abs_diff);
        __m128 pow15_val = _mm_mul_ps(abs_diff, sqrt_val);
        sum = _mm_add_ps(sum, pow15_val);
    }
    
    PORTABLE_ALIGN32 float TmpRes[4];
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float LpDistance_p15_avx_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m256 sum = _mm256_setzero_ps();
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

    while (pVect1 < pEnd1) {
        __m256 v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        __m256 diff = _mm256_sub_ps(v1, v2);
        __m256 abs_diff = _mm256_and_ps(diff, abs_mask);
        __m256 sqrt_val = _mm256_sqrt_ps(abs_diff);
        __m256 pow15_val = _mm256_mul_ps(abs_diff, sqrt_val);
        sum = _mm256_add_ps(sum, pow15_val);
    }
    
    PORTABLE_ALIGN32 float TmpRes[8];
    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

float LpDistance_p15_avx512_16(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    const float* pVect1 = static_cast<const float*>(pVect1v);
    const float* pVect2 = static_cast<const float*>(pVect2v);
    size_t qty = *((const size_t*)qty_ptr);
    const float* pEnd1 = pVect1 + qty;
    __m512 sum = _mm512_setzero_ps();
    
    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1); pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2); pVect2 += 16;
        __m512 diff = _mm512_sub_ps(v1, v2);
        __m512 abs_diff = _mm512_abs_ps(diff);
        __m512 sqrt_val = _mm512_sqrt_ps(abs_diff);
        __m512 pow15_val = _mm512_mul_ps(abs_diff, sqrt_val);
        sum = _mm512_add_ps(sum, pow15_val);
    }
    
    return _mm512_reduce_add_ps(sum);
}

float LpDistance_p15_16_residuals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((const size_t*)qty_ptr);
    size_t qty_simd = qty - (qty % 16);

    float res_simd = L15SqrSIMD16Ext(pVect1v, pVect2v, &qty_simd);
    
    const float* pVect1 = static_cast<const float*>(pVect1v) + qty_simd;
    const float* pVect2 = static_cast<const float*>(pVect2v) + qty_simd;
    size_t qty_left = qty - qty_simd;
    
    float res_scalar = 0;
    for(size_t i=0; i<qty_left; ++i){
        float abs_diff = std::abs(pVect1[i] - pVect2[i]);
        res_scalar += abs_diff * std::sqrt(abs_diff);
    }

    return std::pow(res_simd + res_scalar, 1.0f / 1.5f);
}
#endif

} // namespace hnswlib