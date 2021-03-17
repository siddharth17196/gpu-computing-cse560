#include <iostream>
#include <time.h>
#include <stdio.h>

#define T_WIDTH 16
#define LENGTH 100

using namespace std;

__global__ void transpose(float *a, float *b, int width)
{
    float result = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int k=0; k<width; k++){
        b[k*width + col] = a[row*width + k];
    }
}
__global__ void matmul_shared(float *a, float *b, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    __shared__ float s_a[T_WIDTH][T_WIDTH];
    int row = ty + by*blockDim.y;
    int col = tx + bx*blockDim.x;


    for(int k = 0; k < width/T_WIDTH; ++k)
    {
        s_a[ty][tx] = a[row*width + (k*T_WIDTH + tx)];
        __syncthreads();
        
        for(int p = 0; p < T_WIDTH; ++p){
            b[p][tx] = s_a[ty][p];
        }
        __syncthreads();
    }
}


int main()
{
    static int A[LENGTH][LENGTH];

    int *h_a, *h_trans;
    int *d_a, *d_trans;

    h_a = (int*)malloc(LENGTH*LENGTH*sizeof(int));
	h_trans = (int*)malloc(LENGTH*LENGTH*sizeof(int));

    int k = 0;
    for(int i=0;i<LENGTH;i++)
    {
        for(int j=0;j<LENGTH;j++)
        {
            h_a[k] = A[i][j];
            k+=1;
        }
    }

    cudaMalloc((void**)&d_a, LENGTH*LENGTH*sizeof(int));
	cudaMalloc((void**)&d_trans, LENGTH*LENGTH*sizeof(int));

	cudaMemcpy(d_a, h_a, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

    dim3 dimBlock1(16, 16, 1);
    dim3 dimGrid1(LENGTH/16, LENGTH/16, 1);
	transpose<<<dimGrid1, dimBlock1>>>(d_a, d_trans, LENGTH);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

    // free the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout  << "Time taken (normal) in ms: " << milliseconds << std::endl;
    
	cudaMalloc((void**)&d_a, LENGTH*LENGTH*sizeof(float));
	cudaMalloc((void**)&d_trans, LENGTH*LENGTH*sizeof(float));

	cudaMemcpy(d_a, h_a, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	/* cudaMemcpy(d_b, h_b, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice); */

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    dim3 threads(16, 16, 1);
    dim3 blocks(LENGTH/16, LENGTH/16, 1);
	matmul_shared<<<blocks, threads>>>(d_a, d_trans, LENGTH);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float millisecond = 0;
	cudaEventElapsedTime(&millisecond, start, stop);
	std::cout  << "Time taken (shared memory) in ms: " <<fixed<<millisecond << std::endl;

    return 0;
}
