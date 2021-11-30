typedef unsigned int uint;

uint j_index(uint index){
    return ((index & 15) + (index >> 4)) & 15;
}

__kernel void count_local(__global uint *as, __global uint *count_out, uint power, uint k){

   // 1. Считываем из глобальной памяти в локальную, ставим метку разряда 
   __local uint buf[256];
   uint loc = get_local_id(0);
   uint elem = as[get_global_id(0)];
   buf[loc] = 1 << (((elem >> power) & 3) << 3);
   barrier(CLK_LOCAL_MEM_FENCE);

   // 2. Sum
   uint offset = 0;
   for(int i = 6; i >= 0; i -= 1){
       offset += 1 << i;
       if ((loc & 127) >= offset)
           buf[loc] += buf[loc - (1 << i)];
       barrier(CLK_LOCAL_MEM_FENCE);
   }

   // 3. Записываем счетчик в глобальную память
    if (loc < 4){
        uint pos = loc << 3;
        uint out_index = k * loc + (get_global_id(0) >> 7);

        count_out[out_index] = (buf[127] >> pos) & 127;
        count_out[out_index + 1] = (buf[255] >> pos) & 127;
    }
}

__kernel void count_global_up(__global uint* count_in, __global uint* count_out, uint final){
    // 1. Считываем из глобальной памяти в локальную count'ы (диагональным паттерном!)
    __local uint buf[16][16];
    uint loc = get_local_id(0);
    buf[loc & 15][j_index(loc)] = count_in[get_global_id(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // 2. Проход снизу вверх через for
    for(int i = 2; i < 256 + final; i <<= 1){
        if (((loc + 1) & (i - 1)) == 0){
            uint other_index = loc - (i >> 1);
            uint other_elem = buf[other_index & 15][j_index(other_index)];
            buf[loc & 15][j_index(loc)] += other_elem;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // 3. Запись в count_in И count_out
    if (loc & 1){
        count_in[get_global_id(0)] = buf[loc & 15][j_index(loc)];
        
        if (((loc + 1) & 127) == 0 && !final)
            count_out[get_global_id(0) >> 7] = buf[loc & 15][j_index(loc)];
    }
}

__kernel void count_global_down(__global uint* count_in, __global uint* count_out, uint first){

    __local uint buf[16][16];
    uint loc = get_local_id(0);
    buf[loc & 15][j_index(loc)] = count_out[get_global_id(0)];
    
    if (((loc + 1) & 127) == 0 && !first)
        buf[loc & 15][j_index(loc)] = count_in[get_global_id(0) >> 7];
        
    if (((loc + 1) & 255) == 0 && first)
        buf[loc & 15][j_index(loc)] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Проход сверху вниз через for
    for(int i = 64 * (1 + first); i > 0; i >>= 1){

        // Суммирование
        if (((loc + 1) & ((i << 1) - 1)) == 0){
            uint other_index = loc - i;
            uint other_elem = buf[other_index & 15][j_index(other_index)];
            buf[loc & 15][j_index(loc)] += other_elem;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Копирование
        if (((loc + 1) & (i - 1)) == 0 && ((loc + 1) & ((i << 1) - 1))){
            uint other_index = loc + i;
            uint other_elem = buf[other_index & 15][j_index(other_index)];
            buf[loc & 15][j_index(loc)] = other_elem - buf[loc & 15][j_index(loc)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    count_out[get_global_id(0)] = buf[loc & 15][j_index(loc)];
} 

__kernel void reorder(__global uint* as, __global uint* out, __global uint* count, uint k, uint power){

   // 1a. Считываем из глобальной памяти в локальную массив с настоящими элементами 
   __local uint buf[16][16];
   uint loc = get_local_id(0);
   buf[loc & 15][j_index(loc)] = as[get_global_id(0)];
   barrier(CLK_LOCAL_MEM_FENCE);
   
   // 1b. Заводим массив для скана, ставим метку о принадлежности к какому-то разряду
   __local uint scan[16][16];
   scan[loc & 15][j_index(loc)] =  1 << (((buf[loc & 15][j_index(loc)] >> power) & 3) << 3);
   barrier(CLK_LOCAL_MEM_FENCE);
   
   // 2a. Scan: Up-phase
   for(int i = 2; i < 256; i <<= 1){
        if (((loc + 1) & (i - 1)) == 0){
            uint other_index = loc - (i >> 1);
            uint other_elem = scan[other_index & 15][j_index(other_index)];
            scan[loc & 15][j_index(loc)] += other_elem;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
   }
      
   // 2b. Scan: Down-phase
   if (((loc + 1) & 127) == 0)
       scan[loc & 15][j_index(loc)] = 0;
   barrier(CLK_LOCAL_MEM_FENCE);


   for(int i = 64; i > 0; i >>= 1){

        // Cуммирование
        if (((loc + 1) & ((i << 1) - 1)) == 0){
            uint other_index = loc - i;
            uint other_elem = scan[other_index & 15][j_index(other_index)];
            scan[loc & 15][j_index(loc)] += other_elem;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Копирование
        if (((loc + 1) & (i - 1)) == 0 && ((loc + 1) & ((i << 1) - 1))){
            uint other_index = loc + i;
            uint other_elem = scan[other_index & 15][j_index(other_index)];
            scan[loc & 15][j_index(loc)] = other_elem - scan[loc & 15][j_index(loc)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

   // 3. Считываем из глобальной памяти в локальную offset'ы для данной workgroup'ы
   __local uint offset[4][2];

   if (((loc + 1) & 127) == 0){
       for(int i=0; i<4; i++)
           offset[i][loc >> 7] = count[i * k + (get_global_id(0) >> 7)];
   }
   barrier(CLK_LOCAL_MEM_FENCE);

   // 4. Записываем на основании scan, offset элементы из as в out
   uint num = (buf[loc & 15][j_index(loc)] >> power) & 3;
   uint off = offset[num][loc >> 7];
   uint index = ( scan[loc & 15][j_index(loc)] >> (num << 3) ) & 127;

   out[off + index] = buf[loc & 15][j_index(loc)];
}
