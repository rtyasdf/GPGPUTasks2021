#define WORKGROUP_SIZE 256
typedef unsigned int uint;

__kernel void matrix_transpose(__global float* a, __global float* at, uint K, uint M)
{
    __local float buf[WORKGROUP_SIZE];

    uint globalX = get_global_id(0);
    uint globalY = get_global_id(1);
    uint localX = get_local_id(0);
    uint localY = get_local_id(1);
    uint localSizeX = get_local_size(0);
    uint localSizeY = get_local_size(1);

    uint globalIndex = K * globalY + globalX;
    uint bufIndex = localSizeX * localX + (localX + localY) % localSizeX;

    if (globalX < K && globalY < M)
        buf[bufIndex] = a[globalIndex];
    else
        buf[bufIndex] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    uint newIndex = localSizeX * localY + (localY + localX) % localSizeX;
    
    uint groupOffset = M * (globalX / localSizeX) * localSizeX; 
    uint offsetY =  M * localY;
    uint offsetX = (globalY / localSizeY) * localSizeY;
    uint transposedIndex = groupOffset + offsetY + offsetX + localX;

    if (globalX + (localY - localX) < K && globalY + (localX - localY) < M)
        at[transposedIndex] = buf[newIndex];
}
