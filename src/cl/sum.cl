#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global const unsigned int* a,
                  __global unsigned int* res)
{
    const unsigned int local_index = get_local_id(0);
    const unsigned int size = get_local_size(0);
    unsigned int index = get_global_id(0);
    
    atomic_add(&res[local_index], a[index]);
}
