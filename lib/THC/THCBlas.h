#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

/* Level 1 */
THC_API void THCudaBlas_swap(THCState *state, int64 n, float *x, int64 incx, float *y, int64 incy);
THC_API void THCudaBlas_scal(THCState *state, int64 n, float a, float *x, int64 incx);
THC_API void THCudaBlas_copy(THCState *state, int64 n, float *x, int64 incx, float *y, int64 incy);
THC_API void THCudaBlas_axpy(THCState *state, int64 n, float a, float *x, int64 incx, float *y, int64 incy);
THC_API float THCudaBlas_dot(THCState *state, int64 n, float *x, int64 incx, float *y, int64 incy);

/* Level 2 */
THC_API void THCudaBlas_gemv(THCState *state, char trans, int64 m, int64 n, float alpha, float *a, int64 lda, float *x, int64 incx, float beta, float *y, int64 incy);
THC_API void THCudaBlas_ger(THCState *state, int64 m, int64 n, float alpha, float *x, int64 incx, float *y, int64 incy, float *a, int64 lda);

/* Level 3 */
THC_API void THCudaBlas_gemm(THCState *state, char transa, char transb, int64 m, int64 n, int64 k, float alpha, float *a, int64 lda, float *b, int64 ldb, float beta, float *c, int64 ldc);
THC_API void THCudaBlas_gemmBatched(THCState *state, char transa, char transb, int64 m, int64 n, int64 k,
                                    float alpha, const float *a[], int64 lda, const float *b[], int64 ldb,
                                    float beta, float *c[], int64 ldc, int64 batchCount);

#endif
