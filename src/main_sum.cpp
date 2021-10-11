#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include"cl/sum_cl.h"
#include<vector>
#include<iostream>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        
        ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum");
        sum.compile();

        unsigned int workGroupSize = 128;

        std::vector<unsigned int> res(1, 0);
        
        while (as.size() % workGroupSize > 0) // дополняем до кратности размера рабочей группы
            as.push_back(0);

        unsigned int global_work_size = as.size();

        // создаем буфферы на устройстве
        gpu::gpu_mem_32u as_gpu, res_gpu;
        as_gpu.resizeN(as.size());
        res_gpu.resizeN(1);
        
        as_gpu.writeN(as.data(), as.size());

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            
            // очищаем res_gpu
            res_gpu.writeN(res.data(), 1);
            
            // исполняем kernel
            sum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                     as_gpu, res_gpu);
            t.nextLap();
        }
        
        // изменяем в зависимости от типа устройства вывод
        cl_device_type deviceType;
        clGetDeviceInfo(device.device_id_opencl, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, nullptr);
        std::string deviceString;
        switch(deviceType){
            case 2:
                deviceString = "CPU";
                break;
            case 4:
                deviceString = "GPU";
                break;
            default:
                return 1;
        }
        std::cout << deviceString + ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << deviceString + ":     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl; 

        // считываем в память host'а
        res_gpu.readN(res.data(), 1);
        
        // Проверяем корректность результатов
        EXPECT_THE_SAME(reference_sum, res[0], "GPU results should be equal to CPU results!");
    }

    return 0;
}
