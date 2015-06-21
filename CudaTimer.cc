#include "CudaTimer.h"
#include <cstdio>
#include <cstdlib>

#define SafeTimerCall(err) __safeTimerCall(err, __FILE__, __LINE__)

inline void __safeTimerCall(cudaError err, const char *file, const int line) {
#pragma warning(push)
#pragma warning(disable: 4127) Prevent warning on do-while(0);
  do {
    if (cudaSuccess != err) {
      fprintf(stderr, "CudaTimer failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
      exit(-1);
    }
  } while (0);
#pragma warning(pop)
  return;
}

CudaTimer::CudaTimer() {
  SafeTimerCall(cudaEventCreate(&_begEvent));
  SafeTimerCall(cudaEventCreate(&_endEvent));
  return;
}

CudaTimer::~CudaTimer() {
  SafeTimerCall(cudaEventDestroy(_begEvent));
  SafeTimerCall(cudaEventDestroy(_endEvent));
  return;
}

void CudaTimer::start() {
  SafeTimerCall(cudaEventRecord(_begEvent, 0));
  return;
}

void CudaTimer::stop() {
  SafeTimerCall(cudaEventRecord(_endEvent, 0));
  return;
}

float CudaTimer::value() {
  SafeTimerCall(cudaEventSynchronize(_endEvent));
  float timeVal;
  SafeTimerCall(cudaEventElapsedTime(&timeVal, _begEvent, _endEvent));
  return timeVal;
}
