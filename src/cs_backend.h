#ifndef CS_BACKEND_INTERNAL_H
#define CS_BACKEND_INTERNAL_H

#include "codec_stereo/cs.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A factory pairs a backend's ops table with a constructor for its opaque
 * context. Backends are registered statically in cs_core.c behind
 * compile-time CS_ENABLE_* switches (set by CMake) -- no dynamic/ctor-based
 * registration, to keep link order and static-init order irrelevant.
 */
typedef struct cs_backend_factory {
    void *(*create)(void);   /* allocate + zero-init backend context, or NULL on OOM */
    cs_backend_ops ops;
    /* Static copy of the backend's extraction mode, used to order auto-probe
       attempts without needing a live context (get_caps() requires one). */
    cs_extract_mode mode;
} cs_backend_factory;

#ifdef CS_ENABLE_REF_SAD
cs_backend_factory cs_backend_ref_sad_factory(void);
#endif

#ifdef CS_ENABLE_LAVC
cs_backend_factory cs_backend_lavc_sw_factory(void);
#endif

#ifdef CS_ENABLE_RKMPP
cs_backend_factory cs_backend_rkmpp_factory(void);
#endif

#ifdef CS_ENABLE_RKMPP_HWENC
cs_backend_factory cs_backend_rkmpp_hwenc_factory(void);
#endif

#ifdef CS_ENABLE_D3D12
cs_backend_factory cs_backend_d3d12_vme_factory(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* CS_BACKEND_INTERNAL_H */
