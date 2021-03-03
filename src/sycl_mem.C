#include <CL/sycl.hpp>
#include <cstddef>

#include "sycl_mem.h"

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
