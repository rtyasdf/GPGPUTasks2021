#define WORKGROUP_SIZE 256

__kernel void matrix_transpose(__global float* a, __global float* at, unsigned int K, unsigned int M)
{
    __local float buf[WORKGROUP_SIZE];

    unsigned int globalX = get_global_id(0);
    unsigned int globalY = get_global_id(1);
    unsigned int localX = get_local_id(0);
    unsigned int localY = get_local_id(1);
    unsigned int localSizeX = get_local_size(0);
    unsigned int localSizeY = get_local_size(1);

    unsigned int globalIndex = K * globalY + globalX;
    unsigned int bufIndex = localSizeX * localX + (localX + localY) % localSizeX;

    if (globalX < K && globalY < M)
        buf[bufIndex] = a[globalIndex];
    else
        buf[bufIndex] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int newIndex = localSizeX * localY + (localY + localX) % localSizeX;
    
    unsigned int groupOffset = M * (globalX / localSizeX) * localSizeX; 
    unsigned int offsetY =  M * localY;
    unsigned int offsetX = (globalY / localSizeY) * localSizeY;
    unsigned int transposedIndex = groupOffset + offsetY + offsetX + localX;

    if (globalX + (localY - localX) < K && globalY + (localX - localY) < M)
        at[transposedIndex] = buf[newIndex];
}
