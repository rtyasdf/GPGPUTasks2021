#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
#define SUBGROUP_SIZE 16

__kernel void sum(__global const unsigned int* a,
                  __global unsigned int* res)
{
    const unsigned int local_index = get_local_id(0);
    const unsigned int size = get_local_size(0);
    unsigned int accumulator = 0;
    
    __local unsigned int buffer[WORKGROUP_SIZE];
    buffer[local_index] = a[get_global_id(0)];
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index < SUBGROUP_SIZE){
        for(int i=local_index; i < size; i += SUBGROUP_SIZE)
            accumulator += buffer[i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    buffer[local_index] = accumulator;    
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_index == 0){
        for(int i=1; i < SUBGROUP_SIZE; i++)
            accumulator += buffer[i];
        atomic_add(res, accumulator);
    }
}
