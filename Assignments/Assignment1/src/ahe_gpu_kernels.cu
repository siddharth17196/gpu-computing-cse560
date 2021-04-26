// Implement kernels here (Note: delete sample code below)

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include "ahe_gpu.h"
#include "defines.h"

__constant__ unsigned char cmapping[10*10*256];


__global__ void Equal(unsigned char *img, int *pdf, int nx, int ny, int width){

	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int c = img[i];
	int y = i/width;
	int x = i%width;
	int tx = x/TILE_SIZE_X;
	int ty = y/TILE_SIZE_Y;
	
	// calcuate pdf
	atomicAdd(&pdf[(nx*ty+tx)*256+c], 1);
	__syncthreads();
}

__global__ void cdfmapp(int* cdf, int* pdf, unsigned char *mappings, int ppt){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int c0;
	if (i%256==0)
		c0 = pdf[i];
	__syncthreads();
	int val = c0;
	for(int p=1;p <= threadIdx.x; p++){
		val += pdf[blockDim.x*blockIdx.x+p]; 
	}
	cdf[i] = val;
	__syncthreads();

	mappings[i] =  (unsigned char)round(255.0 * float(val - c0)/float(ppt - c0));
}

__global__ void Interpolate(unsigned char *img, unsigned char *img_out, int nx, int ny, int width)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int y = i/width;
	int x = i%width;
	int c = img[i];
	int x0 = (x - TILE_SIZE_X/2)/TILE_SIZE_X;
	int y0 = (y - TILE_SIZE_Y/2)/TILE_SIZE_Y;
	int x1 = (x + TILE_SIZE_X/2)/TILE_SIZE_X;
	int y1 = (y + TILE_SIZE_Y/2)/TILE_SIZE_Y;
	if(x0 < 0)
		x0 = 0;
	if(y0 < 0)
		y0 = 0;
	if(x1 >= nx)
		x1 = nx-1;
	if(y1 >= ny)
		y1 = ny-1;

	//  interp function
	unsigned char v00, v01, v10, v11;
	v00 = cmapping[c + 256*(x0 + y0*nx)];
	v01 = cmapping[c + 256*(x0 + y1*nx)];
	v10 = cmapping[c + 256*(x1 + y0*nx)];
	v11 = cmapping[c + 256*(x1 + y1*nx)];
	float x_frac = float(x - x0*TILE_SIZE_X - TILE_SIZE_X/2)/float(TILE_SIZE_X);
	float y_frac = float(y - y0*TILE_SIZE_Y - TILE_SIZE_Y/2)/float(TILE_SIZE_Y);
	
	float v0 = v00*(1 - x_frac) + v10*x_frac;
	float v1 = v01*(1 - x_frac) + v11*x_frac;
  	float v = v0*(1 - y_frac) + v1*y_frac;

	if (v < 0) v = 0;
	if (v > 255) v = 255;

	img_out[i] = (unsigned char)(v);
}


extern "C" void aheKernel(unsigned char* img_in, unsigned char* img_out, int width, int height)
{
	int pixels_per_tile = TILE_SIZE_X*TILE_SIZE_Y;
	int ntiles_x = width / TILE_SIZE_X;
	int ntiles_y = height / TILE_SIZE_Y;

	int *d_pdf, *d_cdf;
	unsigned char *d_img;
	unsigned char *d_mapping, *d_out;

	//to measure kernel time
	cudaEvent_t s1, e1;
	float msecs, total_time=0.0;


	cudaMalloc((void**)&d_pdf, ntiles_x*ntiles_y*256*sizeof(int));
	cudaMalloc((void**)&d_cdf, ntiles_x*ntiles_y*256*sizeof(int));
	cudaMalloc((void**)&d_mapping, ntiles_x*ntiles_y*256*sizeof(unsigned char));
	cudaMalloc((void**)&d_img, height*width*sizeof(unsigned char));

	// image from host to device
	cudaMemcpy(d_img, img_in, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);

	cudaEventCreate(&s1);
	cudaEventCreate(&e1);
	cudaEventRecord(s1, 0);
	//kernel call for equalization
	Equal<<<(height*width)/256, 256>>>(d_img, d_pdf, ntiles_x, ntiles_y, width);
	cudaDeviceSynchronize();
	cudaEventRecord(e1, 0);
	cudaEventSynchronize(e1);
	cudaEventElapsedTime(&msecs, s1, e1);
	
	total_time += msecs;

	cudaEventRecord(s1, 0);
	cdfmapp<<<ntiles_y*ntiles_x, 256 >>>(d_cdf, d_pdf, d_mapping, pixels_per_tile);
	cudaDeviceSynchronize();
	cudaEventRecord(e1, 0);
	cudaEventSynchronize(e1);
	cudaEventElapsedTime(&msecs, s1, e1);

	total_time += msecs;

	//store mappings in constant memory
	cudaMemcpyToSymbol(cmapping, d_mapping, ntiles_y*ntiles_x*256*sizeof(unsigned char), 0, cudaMemcpyDeviceToDevice);
	//copy image output to device
	cudaMalloc((void**)&d_out, height*width*sizeof(unsigned char));

	cudaEventRecord(s1, 0);
	//kernel call for interpolation
	Interpolate<<<(height*width)/256, 256>>>(d_img, d_out, ntiles_x, ntiles_y, width);
	cudaDeviceSynchronize();
	cudaEventRecord(e1, 0);
	cudaEventSynchronize(e1);
	cudaEventElapsedTime(&msecs, s1, e1);

	total_time += msecs;
	std::cout<<"Kernel Time: "<<total_time<<" milliseconds\n";


	//get the image output from device
	cudaMemcpy(img_out, d_out, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}
