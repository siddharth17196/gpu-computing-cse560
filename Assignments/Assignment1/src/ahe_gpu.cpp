#include "ahe_gpu.h"
#include <cuda_runtime.h>

#include <iostream>

extern "C" void aheKernel();

void adaptiveEqualizationGPU(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
  aheKernel(); // Remove me!
}
