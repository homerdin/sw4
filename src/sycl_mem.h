#ifndef __SYCL_MEM_H__
#define __SYCL_MEM_H__

#include "Mspace.h"

class QU {
  public:
    static cl::sycl::queue* qu;
//    static umpire::Allocator* allocator;
    static camp::resources::Resource sycl_res;
};

//cl::sycl::queue* QU::qu;

bool syclMallocShared(void* addr, std::size_t size); //{
//  addr = cl::sycl::malloc_shared( size, *QU::qu);
  //return true;
//}

bool syclMallocDevice(void* addr, std::size_t size); // {
  //addr = cl::sycl::malloc_device( size, *QU::qu);
  //return true;
//}

bool syclMallocHost(void* addr, std::size_t size);// {
//  addr = cl::sycl::malloc_host( size, *QU::qu);
  //return true;
//}

#endif
