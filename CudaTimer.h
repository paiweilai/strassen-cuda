#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>

class CudaTimer
{
private:
    cudaEvent_t _begEvent;
    cudaEvent_t _endEvent;

public:
    CudaTimer();
    ~CudaTimer();
    void start();
    void stop();
    float value();
};

#endif // CUDA_TIMER_H
