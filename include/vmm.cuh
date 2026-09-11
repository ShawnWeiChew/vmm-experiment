#pragma once

#include "util.cuh"

static constexpr CUmemAllocationHandleType HANDLE_TYPE = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
typedef CUmemGenericAllocationHandle handle;

/**
 * create a memory handle. A handle seems to be some sort of object that manages the memory
 * allocation
 */
__host__ inline static void vm_alloc(handle* handle,
                                     size_t* allocated_size,
                                     const size_t size,
                                     const int device_id) {
    CUmemAllocationProp prop{};
    prop.location.id = device_id;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = HANDLE_TYPE;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;

    size_t granularity;
    MKERNEL_CUCHECK(
        cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    *allocated_size = (*allocated_size + granularity - 1) / granularity * granularity;

    MKERNEL_CUCHECK(cuMemCreate(handle, *allocated_size, &prop, 0));
}

/**
 * Allocate memory and then assign handle to it
 */
__host__ inline static void vm_map(void** ptr, const handle& handle, const size_t size) {
    CUdeviceptr device_ptr;
    MKERNEL_CUCHECK(cuMemAddressReserve(&device_ptr, size, 0, 0, 0));
    MKERNEL_CUCHECK(cuMemMap(device_ptr, size, 0, handle, 0));
    *ptr = (void*)device_ptr;
}

/**
 */

// __host__ bool check_vmm_suppport(int device_id) {
//     CUdevice dev;
//     MKERNEL_CUCHECK(cuDeviceGet(&dev, device_id));

//     int fd_handle_supported = 0;
//     MKERNEL_CUCHECK(
//         cuDeviceGetAttribute(&fd_handle_supported,
//                              CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
//                              device_id));

//     return fd_handle_supported;
// }

// __host__ inline void export_handle() {

// }