#include <iostream>
#include <stdio.h>
#include <time.h>

#define LENGTH 1000
#define T_WIDTH 16
using namespace std;


__global__ void matmul_shared(float *a, float *b, float *c, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    __shared__ float s_a[T_WIDTH][T_WIDTH];
    __shared__ float s_b[T_WIDTH][T_WIDTH];
    int row = ty + by*blockDim.y;
    int col = tx + bx*blockDim.x;

    float result = 0;

    for(int k = 0; k < width/T_WIDTH; ++k)
    {
        s_a[ty][tx] = a[row*width + (k*T_WIDTH + tx)];
        s_b[ty][tx] = b[(k*T_WIDTH + ty)*width + col];
        __syncthreads();
        
        for(int p = 0; p < T_WIDTH; ++p){
            result += s_a[ty][p] * s_b[p][tx];
        }
        __syncthreads();
    }
    c[row*width + col] = result;
}


__global__ void matmul(float *a, float *b, float *c, int width)
{
    float result = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int k=0; k<width; k++){
        result += a[row*width + k]*b[k*width + col];
    }
    c[row*width + col] = result;
}


int main(){

	static float a_vec[LENGTH][LENGTH];
	static float b_vec[LENGTH][LENGTH];
	/* static float h_c[LENGTH][LENGTH]; */

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    

	h_a = (float*)malloc(LENGTH*LENGTH*sizeof(float));
	h_b = (float*)malloc(LENGTH*LENGTH*sizeof(float));
	h_c = (float*)malloc(LENGTH*LENGTH*sizeof(float));

	for(int i=0 ; i< LENGTH; i++)
    {
        for(int j=0;j<LENGTH;j++){
		    a_vec[i][j] = 1;
		    b_vec[i][j] = 1;
	    }
    }
    int k = 0;
    for(int i=0;i<LENGTH;i++)
    {
        for(int j=0;j<LENGTH;j++)
        {
            h_a[k] = a_vec[i][j];
            h_b[k] = b_vec[i][j];
            k+=1;
        }
    }

    
	cudaMalloc((void**)&d_a, LENGTH*LENGTH*sizeof(float));
	cudaMalloc((void**)&d_b, LENGTH*LENGTH*sizeof(float));
	cudaMalloc((void**)&d_c, LENGTH*LENGTH*sizeof(float));

	cudaMemcpy(d_a, h_a, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

    dim3 dimBlock1(16, 16, 1);
    dim3 dimGrid1(LENGTH/16, LENGTH/16, 1);
	matmul<<<dimGrid1, dimBlock1>>>(d_a, d_b, d_c, LENGTH);

	/* matmul<<<LENGTH*LENGTH/256, 256>>>(d_a, d_b, d_c, LENGTH); */

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
	cudaMalloc((void**)&d_b, LENGTH*LENGTH*sizeof(float));
	cudaMalloc((void**)&d_c, LENGTH*LENGTH*sizeof(float));

	cudaMemcpy(d_a, h_a, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, LENGTH*LENGTH*sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    dim3 threads(16, 16, 1);
    dim3 blocks(LENGTH/16, LENGTH/16, 1);
	matmul_shared<<<blocks, threads>>>(d_a, d_b, d_c, LENGTH);
	/* matmul_shared<<<LENGTH*LENGTH/100, 100>>>(d_a, d_b, d_c, LENGTH); */

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	/* cudaMemcpy(h_c, d_c, LENGTH*LENGTH*sizeof(float), cudaMemcpyDeviceToHost); */
    
    /* for(int i=11000;i<11100;i++){ */
        /* std::cout<<h_c[i]<<std::endl; */
    /* } */

    // free the memory
    /* cudaFree(d_a); */
    /* cudaFree(d_b); */
    /* cudaFree(d_c); */

    /* free(milliseconds); */

	float millisecond = 0;
	cudaEventElapsedTime(&millisecond, start, stop);
	std::cout  << "Time taken (shared memory) in ms: " <<fixed<<millisecond << std::endl;
}
