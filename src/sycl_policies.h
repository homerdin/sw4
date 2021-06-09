#ifndef __SYCL_POLICIES_H__
#define __SYCL_POLICIES_H__
#include "RAJA/RAJA.hpp"
#include "camp/resource.hpp"
#include "Mspace.h"
#include "sycl_mem.h"
/*class QU {
  public:
    static cl::sycl::queue* qu;
//    static umpire::Allocator* allocator;
};*/

//cl::sycl::queue* QU::qu;
/*
bool syclMallocShared(void* addr, std::size_t size) {
  addr = cl::sycl::malloc_shared( size, *QU::qu);
  return true;
}

bool syclMallocDevice(void* addr, std::size_t size) {
  addr = cl::sycl::malloc_device( size, *QU::qu);
  return true;
}

bool syclMallocHost(void* addr, std::size_t size) {
  addr = cl::sycl::malloc_host( size, *QU::qu);
  return true;
}
*/
#define SW4_FORCEINLINE // always_inline
#define SYNC_DEVICE (*QU::qu).wait(); //SW4_CheckDeviceError(syclDeviceSynchronize())
#define SYNC_STREAM QU::qu->wait();//SW4_CheckDeviceError(syclStreamSynchronize(0))
#define SW4_PEEK //SW4_CheckDeviceError(syclPeekAtLastError());

#define RAJA_HOST_DEVICE SYCL_EXTERNAL

#define SW4_MALLOC_MANAGED(addr, size) (syclMallocShared(addr, size))
#define SW4_MALLOC_DEVICE(addr, size) (syclMallocDevice(addr, size))
#define SW4_MALLOC_PINNED(addr, size) (syclMallocHost(addr, size))

#define SW4_FREE_MANAGED(addr) (sycl::free(addr, *QU::qu));
#define SW4_FREE_DEVICE(addr) (sycl::free(addr, *QU::qu));
#define SW4_FREE_PINNED(addr) (sycl::free(addr, *QU::qu));

#define SW4_DEVICE_SUCCESS true 
//   SW4_CheckDeviceError(syclStreamSynchronize(0));
typedef RAJA::sycl_exec<256, false> DEFAULT_LOOP1;
typedef RAJA::sycl_exec<256, false> DEFAULT_LOOP1_ASYNC;
using REDUCTION_POLICY = RAJA::sycl_reduce;

typedef RAJA::sycl_exec<256, false> PREDFORT_LOOP_POL;
typedef RAJA::sycl_exec<256, false> PREDFORT_LOOP_POL_ASYNC;

typedef RAJA::sycl_exec<256, false> CORRFORT_LOOP_POL;
typedef RAJA::sycl_exec<256, false> CORRFORT_LOOP_POL_ASYNC;

typedef RAJA::sycl_exec<256, false> DPDMTFORT_LOOP_POL;

typedef RAJA::sycl_exec<256, false> DPDMTFORT_LOOP_POL_ASYNC;

typedef RAJA::sycl_exec<256, false> SARRAY_LOOP_POL1;

using DEFAULT_LOOP2X = //RAJA::KernelPolicy<RAJA::statement::SyclKernel<
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

/*RAJA::statement::For<1, RAJA::sycl_group_1_loop,
                         RAJA::statement::For<0, RAJA::sycl_local_1_loop,
                                              RAJA::statement::Lambda<0>>>>>;
*/
using DEFAULT_LOOP2X_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
/*RAJA::KernelPolicy<RAJA::statement::SyclKernel<RAJA::statement::For<
        1, RAJA::sycl_group_1_loop,
        RAJA::statement::For<0, RAJA::sycl_local_1_loop,
                             RAJA::statement::Lambda<0>>>>>;
*/
using DEFAULT_LOOP3 =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using COPY_KPLANE_EXEC_POL = DEFAULT_LOOP3;

using DPDMT_WIND_LOOP_POL_ASYNC = 
  RAJA::KernelPolicy<
    RAJA::statement::SyclKernel<
      RAJA::statement::For< 3, RAJA::sycl_global_0<1>,
        RAJA::statement::For< 2, RAJA::seq_exec,
          RAJA::statement::For< 1, RAJA::seq_exec,
            RAJA::statement::For<0, RAJA::seq_exec,
              RAJA::statement::Lambda<0>
	    >
	  >
	>
      >
    >
  >;

using SARRAY_LOOP_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<1>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<256>, // i
              RAJA::statement::For<3, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;



using ICSTRESS_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using ICSTRESS_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernelNonTrivial<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using RHS4_EXEC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernelNonTrivial<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using RHS4_EXEC_POL_ASYNC_OLDE =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>    // j

          >
        >
      >
    >;

using RHS4_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernelNonTrivial<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using CONSINTP_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;


using ODDIODDJ_EXEC_POL1_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernelNonTrivial<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          > 
        >   
      >         
    >;

using ODDIODDJ_EXEC_POL2_ASYNC = RHS4_EXEC_POL_ASYNC;

using EVENIODDJ_EXEC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernelNonTrivial<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

/*RAJA::KernelPolicy<RAJA::statement::SyclKernel<RAJA::statement::For<
        1, RAJA::sycl_group_1_loop,
        RAJA::statement::For<0, RAJA::sycl_local_1_loop,
                             RAJA::statement::Lambda<0>>>>>;
*/
using EVENIEVENJ_EXEC_POL_ASYNC = EVENIODDJ_EXEC_POL_ASYNC;

using ODDIEVENJ_EXEC_POL1_ASYNC = ICSTRESS_EXEC_POL_ASYNC;

using ODDIEVENJ_EXEC_POL2_ASYNC = RHS4_EXEC_POL_ASYNC;

using XRHS_POL =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
/*RAJA::KernelPolicy<RAJA::statement::SyclKernel<RAJA::statement::For<
        0, RAJA::sycl_group_1_direct,
        RAJA::statement::For<
            1, RAJA::sycl_group_2_direct,
            RAJA::statement::For<2, RAJA::sycl_group_3_direct,
                                 RAJA::statement::Lambda<0>>>>>>;
*/
using XRHS_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

/*RAJA::KernelPolicy<RAJA::statement::SyclKernelAsync<RAJA::statement::For<
        0, RAJA::sycl_group_1_loop,
        RAJA::statement::For<
            1, RAJA::sycl_group_2_loop,
            RAJA::statement::For<2, RAJA::sycl_local_1_loop,
                                 RAJA::statement::Lambda<0>>>>>>;
*/
using TWILIGHTSG_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using CONSINTP_EXEC_POL4 = ICSTRESS_EXEC_POL;

using CONSINTP_EXEC_POL5 = ICSTRESS_EXEC_POL;

using PRELIM_CORR_EXEC_POL1 = DEFAULT_LOOP2X;
using PRELIM_CORR_EXEC_POL1_ASYNC = DEFAULT_LOOP2X_ASYNC;

using PRELIM_PRED_EXEC_POL1 = ICSTRESS_EXEC_POL;
using PRELIM_PRED_EXEC_POL1_ASYNC = ICSTRESS_EXEC_POL_ASYNC;

// using ENFORCEBC_CORR_EXEC_POL1 =  ICSTRESS_EXEC_POL;

using ENFORCEBC_CORR_EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using BCFORT_EXEC_POL1 = RHS4_EXEC_POL;
using BCFORT_EXEC_POL2 = ICSTRESS_EXEC_POL;

using ENERGY4CI_EXEC_POL = RHS4_EXEC_POL;


// Next 4 in solve.C
using DHI_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
/*
RAJA::KernelPolicy<RAJA::statement::SyclKernelAsync<RAJA::statement::For<
        0, RAJA::sycl_group_1_loop,
        RAJA::statement::For<
            1, RAJA::sycl_group_2_loop,
            RAJA::statement::For<2, RAJA::sycl_local_1_loop,
                                 RAJA::statement::Lambda<0>>>>>>;
*/
using GIG_POL_ASYNC =// RAJA::KernelPolicy<RAJA::statement::SyclKernelAsync<
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
/*
RAJA::statement::For<1, RAJA::sycl_group_1_loop,
                         RAJA::statement::For<0, RAJA::sycl_local_1_loop,
                                              RAJA::statement::Lambda<0>>>>>;
*/
using AVS_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using EBFA_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

// void EW::get_exact_point_source in EW.C
using GEPS_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_1<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_0<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

// CurvilinearInterface2::bnd_zero
using BZ_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

// CurvilinearInterface2::injection
using INJ_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;


// TODO: Below
using INJ_POL2_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

/*RAJA::KernelPolicy<RAJA::statement::SyclKernel<RAJA::statement::For<
        1, RAJA::sycl_group_1_loop,
        RAJA::statement::For<0, RAJA::sycl_local_1_loop,
                             RAJA::statement::Lambda<0>>>>>;
*/
// CurvilinearInterface2::communicate_array

using CA_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;



// Sarray::assign

using SAA_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_1<1>,      // k
          RAJA::statement::For<3, RAJA::sycl_global_0<16>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<4>, // i
              RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;

// Sarray::insert_intersection(
using SII_POL =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;


// TestEcons::get_ubnd(
using TGU_POL_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
// TODO BELOW

// in addmemvarforcing2.C
using AMVPCa_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;
 

using AMVPCu_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using AMVC2Ca_POL_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using AMVC2Cu_POL =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

// In addsg4windc.C
using ASG4WC_POL_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<2, RAJA::sycl_global_0<1>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
            RAJA::statement::For<0, RAJA::sycl_global_2<1>, // ii
	      RAJA::statement::For<3, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >
    >;


// In addsgdc.C
using ADDSGD_POL_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_1<4>,      // k
          RAJA::statement::For<3, RAJA::sycl_global_0<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<16>, // i
	      RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
	      >
            >
          >
        >
      >
    >;


using ADDSGD_POL2_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_1<4>,      // k
          RAJA::statement::For<3, RAJA::sycl_global_0<16>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_2<4>, // i
	      RAJA::statement::For<0, RAJA::seq_exec,
                RAJA::statement::Lambda<0>
	      >
            >
          >
        >
      >
    >;


// in bcforce.C
using BCFORT_EXEC_POL2_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_1<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_2<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using BCFORT_EXEC_POL3_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;


// in curvilinear4sgc.C
using CURV_POL_ORG =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<1>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using CURV_POL = DEFAULT_LOOP3;
// in parallelStuff.C
using BUFFER_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_global_0<4>,      // k
          RAJA::statement::For<0, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

/*RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<1, RAJA::sycl_group_1_loop,      // k
          RAJA::statement::For<0, RAJA::sycl_local_1_loop,
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;
*/
// in rhs3cuvilinearsgc.C
using RHS4CU_POL_ASYNC =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

// in rhs4th3fortc.C
using XRHS_POL2 =
     RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<1>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

// ^^^ TODO ^^^
using RHS4TH3_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using RHS4TH3_POL2_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_2<4>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<4>,    // j
            RAJA::statement::For<2, RAJA::sycl_global_0<16>, // i
              RAJA::statement::Lambda<0>
            >
          >
        >
      >
    >;

using VBSC_POL =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<1>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

using AFCC_POL_ASYNC =
    RAJA::KernelPolicy<
      RAJA::statement::SyclKernel<
        RAJA::statement::For<0, RAJA::sycl_global_0<16>,      // k
          RAJA::statement::For<1, RAJA::sycl_global_1<16>,    // j
            RAJA::statement::Lambda<0>
          >
        >
      >
    >;

// In updatememvarc.C
using MPFC_POL_ASYNC = DEFAULT_LOOP3;

// IN EW.C
using FORCE_LOOP_ASYNC = RAJA::sycl_exec<32, false>;
using FORCETT_LOOP_ASYNC = RAJA::sycl_exec<256, false>;
#endif
