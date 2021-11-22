#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000. / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Массив оригинальных данных
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    // Массив - копия
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    std::vector<gpu::gpu_mem_32u> data;
    data.push_back(as_gpu);
    data.push_back(bs_gpu);

    // Вектор счетчиков и вектор их размеров
    std::vector<unsigned int> csize(1, n / 32);
    std::vector<gpu::gpu_mem_32u> count;
    
    while(csize.back() != 256){
        // Добавляем счетчик
        gpu::gpu_mem_32u level_count;
        level_count.resizeN(csize.back());
        count.push_back(level_count);
        
        // Добавляем размер, подогнанный под 256
        int s = csize.back();
        s = (s - 1) / 128 + 1;
        s = ((s - 1) / 256 + 1) * 256;
        csize.push_back(s);
    }
    
    // Финальный счетчик, его размер всегда 256
    gpu::gpu_mem_32u level_count;
    level_count.resizeN(csize.back());
    count.push_back(level_count);

    {
        ocl::Kernel count_local(radix_kernel, radix_kernel_length, "count_local");
        ocl::Kernel count_up(radix_kernel, radix_kernel_length, "count_global_up");
        ocl::Kernel count_down(radix_kernel, radix_kernel_length, "count_global_down");
        ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");

        count_local.compile();
        count_up.compile();
        count_down.compile();
        reorder.compile();

        std::vector<unsigned int> zeros(n / 32, 0);
        timer t;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            data[0].writeN(as.data(), n);
            data[1].writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            // Идём по разрядам
            for(int i=0; i < 32; i+=2){

                int f = (i / 2) % 2;
                int s = (f + 1) % 2;

                // Обнуляем count'ы
                for(int j=0; j<csize.size(); j++)
                    count[j].writeN(zeros.data(), csize[j]);

                unsigned int workGroupSize = 256;
                unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

                // Считаем по рабочей группе, сколько в ней каких разрядов 
                count_local.exec(gpu::WorkSize(workGroupSize, global_work_size), data[f], count[0], i, (n - 1) / 128 + 1);
            
                // Проходим вверх, считая локальные префиксы
                for(int j=1; j <= count.size(); j++)
                    count_up.exec(gpu::WorkSize(256, csize[j-1]), count[j-1], count[j % count.size()], j == count.size() ? 1 : 0);

                // Проходим вниз, считая глобальные префиксы 
                for(int j=count.size(); j > 0; j--)
                    count_down.exec(gpu::WorkSize(256, csize[j-1]), count[j % count.size()], count[j-1], j == count.size() ? 1 : 0);

                // Переупорядочиваем, в соответствии с отступами
                reorder.exec(gpu::WorkSize(256, n), data[f], data[s], count[0], (n - 1) / 128 + 1, i); 
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        data[0].readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
