#pragma once
#include "hnswlib.h"
#include "distcomp.h" // Contains LP distance declarations

namespace hnswlib {

class LPSpace : public SpaceInterface<float> {
    struct LpParams {
        size_t dim;
        float p;
    };

    LpParams params_;
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;

    // Dispatch to optimized implementations based on p
    static float LpDistanceDispatcher(const void* a, const void* b, const void* param) {
        const LpParams* p = reinterpret_cast<const LpParams*>(param);
        
        // Handle special cases with optimized implementations
        if (p->p == 1.0f) {
            return similarity::L1NormSIMD(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                p->dim
            );
        } else if (p->p == 2.0f) {
            return similarity::L2NormSIMD(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                p->dim
            );
        } else if (std::isinf(p->p)) {
            return similarity::LInfNormSIMD(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                p->dim
            );
        } else {
            return similarity::LPGenericDistanceOptim(
                reinterpret_cast<const float*>(a),
                reinterpret_cast<const float*>(b),
                p->dim,
                p->p
            );
        }
    }

public:
    LPSpace(size_t dim, float p) : params_({dim, p}), data_size_(dim * sizeof(float)) {
        fstdistfunc_ = LpDistanceDispatcher;
    }

    size_t get_data_size() override { return data_size_; }
    DISTFUNC<float> get_dist_func() override { return fstdistfunc_; }
    void* get_dist_func_param() override { return &params_; }

    ~LPSpace() {}
};

} // namespace hnswlib