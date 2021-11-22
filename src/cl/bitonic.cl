#define wkgroup_size 128

typedef unsigned int uint;

__kernel void bitonic(__global float *as, uint power, uint local_limit, uint dir_power) {

    __local float buf[2 * wkgroup_size];

    uint g = get_global_id(0);
    uint l = get_local_id(0);

    // Индексы пары в глобальной памяти
    uint p = 1 << power;
    uint low = (g >> power) << (power + 1);
    low += g & (p - 1);
    uint high = low + p;

    // Какой порядок для текущей пары рассматривать (на схеме -- в какую сторону направлена стрелка)
    uint m = (g & ((1 << dir_power) - 1)) >> (dir_power - 1);

    if ( power > local_limit){ // Если не сможем досортировать только в локальной памяти

        // Расположение элементов в локальной памяти не соответствует расположению в глобальной
        // Но на это всё равно, т.к. главное их просто сравнить один раз, а затем вернуть в глобальную  
        buf[l] = as[low];
        buf[l + wkgroup_size] = as[high];

        float d = buf[l] - buf[l + wkgroup_size];

        // Учитываем направление стрелки
        if ((d > 0 && !m) || (d < 0 && m)){
            d = buf[l];
            buf[l] = buf[l + wkgroup_size];
            buf[l + wkgroup_size] = d;
        }
        
        // Возвращение в глобальную память
        as[low] = buf[l];
        as[high] = buf[l + wkgroup_size];
    }
    else{ // Если и дальнейшие итерации можно провести в локальной памяти

        // Подбираем индексы, так чтобы расположение в локальной памяти
        // соответствовало расположению в глобальной
        uint new_low = (l >> power) << (power + 1);
        new_low += l & (p - 1);
        uint new_high = new_low + p;
        
        buf[new_low] = as[low];
        buf[new_high] = as[high];
        
        // Добиваем всеми степенями двойки от 2 ^ power вплоть до 1
        for(int i=power; i >= 0 ; i--){

            // Индексы пар для 2 ^ i
            p = 1 << i;
            new_low = (l >> i) << (i + 1);
            new_low += l & (p - 1);
            new_high = new_low + p;

            float d = buf[new_low] - buf[new_high];

            if ((d > 0 && !m) || (d < 0 && m)){
                d = buf[new_low];
                buf[new_low] = buf[new_high];
                buf[new_high] = d;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Возвращаем в глобальную память
        p = 1 << power;
        new_low = (l >> power) << (power + 1);
        new_low += l & (p - 1);
        new_high = new_low + p;
        
        as[low] = buf[new_low];
        as[high] = buf[new_high];
    }
}
