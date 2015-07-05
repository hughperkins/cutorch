#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"
#include "THCGeneral.h"

#define TH_TENSOR_REFCOUNTED 1

typedef struct THCudaTensor
{
    int64 *size;
    int64 *stride;
    int nDimension;

    THCudaStorage *storage;
    int64 storageOffset;
    int refcount;

    char flag;

} THCudaTensor;


/**** access methods ****/
THC_API THCudaStorage* THCudaTensor_storage(THCState *state, const THCudaTensor *self);
THC_API int64 THCudaTensor_storageOffset(THCState *state, const THCudaTensor *self);
THC_API int THCudaTensor_nDimension(THCState *state, const THCudaTensor *self);
THC_API int64 THCudaTensor_size(THCState *state, const THCudaTensor *self, int dim);
THC_API int64 THCudaTensor_stride(THCState *state, const THCudaTensor *self, int dim);
THC_API THLongStorage *THCudaTensor_newSizeOf(THCState *state, THCudaTensor *self);
THC_API THLongStorage *THCudaTensor_newStrideOf(THCState *state, THCudaTensor *self);
THC_API float *THCudaTensor_data(THCState *state, const THCudaTensor *self);

THC_API void THCudaTensor_setFlag(THCState *state, THCudaTensor *self, const char flag);
THC_API void THCudaTensor_clearFlag(THCState *state, THCudaTensor *self, const char flag);


/**** creation methods ****/
THC_API THCudaTensor *THCudaTensor_new(THCState *state);
THC_API THCudaTensor *THCudaTensor_newWithTensor(THCState *state, THCudaTensor *tensor);
/* stride might be NULL */
THC_API THCudaTensor *THCudaTensor_newWithStorage(THCState *state, THCudaStorage *storage_, int64 storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THC_API THCudaTensor *THCudaTensor_newWithStorage1d(THCState *state, THCudaStorage *storage_, int64 storageOffset_,
                                int64 size0_, int64 stride0_);
THC_API THCudaTensor *THCudaTensor_newWithStorage2d(THCState *state, THCudaStorage *storage_, int64 storageOffset_,
                                int64 size0_, int64 stride0_,
                                int64 size1_, int64 stride1_);
THC_API THCudaTensor *THCudaTensor_newWithStorage3d(THCState *state, THCudaStorage *storage_, int64 storageOffset_,
                                int64 size0_, int64 stride0_,
                                int64 size1_, int64 stride1_,
                                int64 size2_, int64 stride2_);
THC_API THCudaTensor *THCudaTensor_newWithStorage4d(THCState *state, THCudaStorage *storage_, int64 storageOffset_,
                                int64 size0_, int64 stride0_,
                                int64 size1_, int64 stride1_,
                                int64 size2_, int64 stride2_,
                                int64 size3_, int64 stride3_);

/* stride might be NULL */
THC_API THCudaTensor *THCudaTensor_newWithSize(THCState *state, THLongStorage *size_, THLongStorage *stride_);
THC_API THCudaTensor *THCudaTensor_newWithSize1d(THCState *state, int64 size0_);
THC_API THCudaTensor *THCudaTensor_newWithSize2d(THCState *state, int64 size0_, int64 size1_);
THC_API THCudaTensor *THCudaTensor_newWithSize3d(THCState *state, int64 size0_, int64 size1_, int64 size2_);
THC_API THCudaTensor *THCudaTensor_newWithSize4d(THCState *state, int64 size0_, int64 size1_, int64 size2_, int64 size3_);

THC_API THCudaTensor *THCudaTensor_newClone(THCState *state, THCudaTensor *self);
THC_API THCudaTensor *THCudaTensor_newContiguous(THCState *state, THCudaTensor *tensor);
THC_API THCudaTensor *THCudaTensor_newSelect(THCState *state, THCudaTensor *tensor, int dimension_, int64 sliceIndex_);
THC_API THCudaTensor *THCudaTensor_newNarrow(THCState *state, THCudaTensor *tensor, int dimension_, int64 firstIndex_, int64 size_);
THC_API THCudaTensor *THCudaTensor_newTranspose(THCState *state, THCudaTensor *tensor, int dimension1_, int dimension2_);
THC_API THCudaTensor *THCudaTensor_newUnfold(THCState *state, THCudaTensor *tensor, int dimension_, int64 size_, int64 step_);

THC_API void THCudaTensor_resize(THCState *state, THCudaTensor *tensor, THLongStorage *size, THLongStorage *stride);
THC_API void THCudaTensor_resizeAs(THCState *state, THCudaTensor *tensor, THCudaTensor *src);
THC_API void THCudaTensor_resize1d(THCState *state, THCudaTensor *tensor, int64 size0_);
THC_API void THCudaTensor_resize2d(THCState *state, THCudaTensor *tensor, int64 size0_, int64 size1_);
THC_API void THCudaTensor_resize3d(THCState *state, THCudaTensor *tensor, int64 size0_, int64 size1_, int64 size2_);
THC_API void THCudaTensor_resize4d(THCState *state, THCudaTensor *tensor, int64 size0_, int64 size1_, int64 size2_, int64 size3_);
THC_API void THCudaTensor_resize5d(THCState *state, THCudaTensor *tensor, int64 size0_, int64 size1_, int64 size2_, int64 size3_, int64 size4_);
THC_API void THCudaTensor_rawResize(THCState *state, THCudaTensor *self, int nDimension, int64 *size, int64 *stride);

THC_API void THCudaTensor_set(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_setStorage(THCState *state, THCudaTensor *self, THCudaStorage *storage_, int64 storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THC_API void THCudaTensor_setStorage1d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, int64 storageOffset_,
                                    int64 size0_, int64 stride0_);
THC_API void THCudaTensor_setStorage2d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, int64 storageOffset_,
                                    int64 size0_, int64 stride0_,
                                    int64 size1_, int64 stride1_);
THC_API void THCudaTensor_setStorage3d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, int64 storageOffset_,
                                    int64 size0_, int64 stride0_,
                                    int64 size1_, int64 stride1_,
                                    int64 size2_, int64 stride2_);
THC_API void THCudaTensor_setStorage4d(THCState *state, THCudaTensor *self, THCudaStorage *storage_, int64 storageOffset_,
                                    int64 size0_, int64 stride0_,
                                    int64 size1_, int64 stride1_,
                                    int64 size2_, int64 stride2_,
                                    int64 size3_, int64 stride3_);

THC_API void THCudaTensor_narrow(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension_, int64 firstIndex_, int64 size_);
THC_API void THCudaTensor_select(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension_, int64 sliceIndex_);
THC_API void THCudaTensor_transpose(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension1_, int dimension2_);
THC_API void THCudaTensor_unfold(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension_, int64 size_, int64 step_);

THC_API void THCudaTensor_squeeze(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_squeeze1d(THCState *state, THCudaTensor *self, THCudaTensor *src, int dimension_);

THC_API int THCudaTensor_isContiguous(THCState *state, const THCudaTensor *self);
THC_API int THCudaTensor_isSameSizeAs(THCState *state, const THCudaTensor *self, const THCudaTensor *src);
THC_API int64 THCudaTensor_nElement(THCState *state, const THCudaTensor *self);

THC_API void THCudaTensor_retain(THCState *state, THCudaTensor *self);
THC_API void THCudaTensor_free(THCState *state, THCudaTensor *self);
THC_API void THCudaTensor_freeCopyTo(THCState *state, THCudaTensor *self, THCudaTensor *dst);

/* Slow access methods [check everything] */
THC_API void THCudaTensor_set1d(THCState *state, THCudaTensor *tensor, int64 x0, float value);
THC_API void THCudaTensor_set2d(THCState *state, THCudaTensor *tensor, int64 x0, int64 x1, float value);
THC_API void THCudaTensor_set3d(THCState *state, THCudaTensor *tensor, int64 x0, int64 x1, int64 x2, float value);
THC_API void THCudaTensor_set4d(THCState *state, THCudaTensor *tensor, int64 x0, int64 x1, int64 x2, int64 x3, float value);

THC_API float THCudaTensor_get1d(THCState *state, const THCudaTensor *tensor, int64 x0);
THC_API float THCudaTensor_get2d(THCState *state, const THCudaTensor *tensor, int64 x0, int64 x1);
THC_API float THCudaTensor_get3d(THCState *state, const THCudaTensor *tensor, int64 x0, int64 x1, int64 x2);
THC_API float THCudaTensor_get4d(THCState *state, const THCudaTensor *tensor, int64 x0, int64 x1, int64 x2, int64 x3);

/* CUDA-specific functions */
THC_API cudaTextureObject_t THCudaTensor_getTextureObject(THCState *state, THCudaTensor *self);
THC_API int THCudaTensor_getDevice(THCState *state, const THCudaTensor *self);
THC_API int THCudaTensor_checkGPU(THCState *state, unsigned int nTensors, ...);

#endif
