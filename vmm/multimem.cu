#include <vector>

#include "../include/multimem.cuh"
#include "../include/util.cuh"

std::once_flag g_once;
int g_world = 0;
bool g_mc_ok = false;
CUmemGenericAllocationHandle g_mch;

bool create_mc(int world) {
    DKF(cuInit(0));
    for (int d = 0; d < world; ++d) {
        CUdevice dev;
        DKF(cuDeviceGet(&dev, d));
        int ok = 0;
        DKF(cuDeviceGetAttribute(&ok, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
        if (!ok) {
            std::fprintf(stderr, "mInfer SP_MM: GPU %d has no multicast\n", d);
            return false;
        }
    }
    CUmulticastObjectProp mp{};
    mp.numDevices = (unsigned)world;
    mp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    mp.size = kSlab;
    size_t gran = 0;
    DKF(cuMulticastGetGranularity(&gran, &mp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (kSlab % gran) {
        std::fprintf(stderr, "mInfer SP_MM: slab %% granularity (%zu)\n", gran);
        return false;
    }
    DKF(cuMulticastCreate(&g_mch, &mp));
    for (int d = 0; d < world; ++d) {
        CUdevice dev;
        DKF(cuDeviceGet(&dev, d));
        DKF(cuMulticastAddDevice(g_mch, dev));
    }
    return true;
}

bool rank_setup(RankState& r, int device, int world_size) {
    CUmemAllocationProp ap{};
    ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ap.location.id = device;
    ap.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmemGenericAllocationHandle ph;
    DKF(cuMemCreate(&ph, kSlab, &ap, 0));
    DKF(cuMulticastBindMem(g_mch, 0, ph, 0, kSlab, 0));
    std::vector<CUmemAccessDesc> ad(world_size);

    for (int i = 0; i < world_size; i++) {
        ad[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        ad[i].location.id = i;
        ad[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    }

    DKF(cuMemAddressReserve(&r.uc, kSlab, 0, 0, 0));
    DKF(cuMemMap(r.uc, kSlab, 0, ph, 0));
    // grant every other peer access to this device's memory
    DKF(cuMemSetAccess(r.uc, kSlab, ad.data(), world_size));

    CUmemAccessDesc ad_local{};
    ad_local.location = ap.location;
    ad_local.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DKF(cuMemAddressReserve(&r.mc, kSlab, 0, 0, 0));
    DKF(cuMemMap(r.mc, kSlab, 0, g_mch, 0));
    DKF(cuMemSetAccess(r.mc, kSlab, &ad_local, 1));
    CKF(cudaMemset((void*)(r.uc + kFlagOff), 0, 4096));
    return true;
}

int minfer_spmm_init(int rank, int world, int device, size_t data_bytes) {
    if (rank < 0)
        return -1;
    if (device != rank) {
        // create_mc enumerates devices 0..world-1; a remapped rank->device layout would
        // bind the wrong memory. Refuse rather than guess.
        std::fprintf(stderr,
                     "mInfer SP_MM: rank %d on device %d - not the identity map, disabled\n",
                     rank,
                     device);
        return -1;
    }
    std::call_once(g_once, [&] {
        g_world = world;
        g_mc_ok = create_mc(world);
    });
    if (!g_mc_ok || world != g_world)
        return -1;
    RankState& r = g_rs[rank];
    if (!r.mapped) {
        if (!rank_setup(r, device, world))
            return -1;
        r.mapped = true;
    }
    r.ready = true;
    std::printf("[rank %d] SP_MM ready: 512 MB multicast slab, twins at +0/+160 MB\n", rank);
    return 0;
}

constexpr int kMaxCtas = 256;
__device__ __forceinline__ void mm_barrier(unsigned* mc_flags,
                                           const unsigned* uc_flags,
                                           int row,
                                           unsigned target) {
    if (threadIdx.x == 0) {
        unsigned* f = mc_flags + row * kMaxCtas + blockIdx.x;
        asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(f), "r"(1u)
                     : "memory");
        const unsigned* l = uc_flags + row * kMaxCtas + blockIdx.x;
        unsigned v;
        do {
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(l) : "memory");
        } while (v < target);
    }
    __syncthreads();
}