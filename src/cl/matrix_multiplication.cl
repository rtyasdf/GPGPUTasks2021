#define WORKGROUP_SIZE 256
typedef unsigned int uint;

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c, uint M, uint K, uint N)
{
    __local float aCache[WORKGROUP_SIZE];
    __local float bCache[WORKGROUP_SIZE];
    float accumulator = 0;

    const uint localSizeX = get_local_size(0);
    const uint localSizeY = get_local_size(1);
    const uint localX = get_local_id(0);
    const uint localY = get_local_id(1);
    const uint globalX = get_global_id(0);
    const uint globalY = get_global_id(1);

    const uint multOffset = localY * localSizeX;
    const uint cacheIndex = localY * localSizeX + localX;
    uint aOffset = 0;
    uint bOffset = 0;

    while(aOffset < K){

        aCache[cacheIndex] = a[globalY * K + aOffset + localX];
        barrier(CLK_LOCAL_MEM_FENCE);

        bCache[cacheIndex] = b[(localY + bOffset) * N + globalX];
        barrier(CLK_LOCAL_MEM_FENCE);

        uint index = localX;
        for(uint j = 0; j < localSizeY; j++){
            accumulator += bCache[index] * aCache[multOffset + j];
            index += localSizeX;
        }
        aOffset += localSizeX;
        bOffset += localSizeY;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[globalY * get_global_size(0) + globalX] = accumulator;
}
