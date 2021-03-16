#include <iostream>
#include <stdio.h>
#include <time.h>


#define LENGTH 10000
using namespace std;



__global__ void vector_add(float *a, float *b, float *c){
	int index = threadIdx.x + blockDim.x * blockIdx.x; // 101
    c[index] = a[index] + b[index];
}

__host__ void vector_add_cpu(float a[], float b[], float *c){
	for(int i=0 ; i< LENGTH ; i++){
		c[i] = a[i] + b[i];
		// std::cout << c[i] << std::endl;
	}
	
}

int main(){

	float *a_vec;
	float *b_vec;

	

	float *c_vec;
	float *d_a, *d_b, *d_c;
	float *h_c;

	a_vec = (float*)malloc(LENGTH*sizeof(float));
	b_vec = (float*)malloc(LENGTH*sizeof(float));
	h_c = (float*)malloc(LENGTH*sizeof(float));


	for(int i=0 ; i< LENGTH; i++){
		a_vec[i] = i;
		b_vec[i] = i;
	}




	cudaMalloc((void**)&d_a, LENGTH*sizeof(float));
	cudaMalloc((void**)&d_b, LENGTH*sizeof(float));
	cudaMalloc((void**)&d_c, LENGTH*sizeof(float)); // host -> device
	
	c_vec = (float*)malloc(LENGTH*sizeof(float)); //cpu device -> host

	cudaMemcpy(d_a, a_vec, LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b_vec, LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	vector_add<<<LENGTH/10, 10>>>(d_a, d_b, d_c);

	cudaDeviceSynchronize();

	cudaEventRecord(stop);

	
	cudaMemcpy(c_vec, d_c, LENGTH*sizeof(float), cudaMemcpyDeviceToHost);
	
    /* for(int i=0; i<100;i++){ */
    /*     std::cout<<c_vec[i]<<std::endl; */
    /* } */
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);


	std::cout  << "Time taken : " << milliseconds << std::endl;
	for(int i=0; i<LENGTH ;i++){
		cout << c_vec[i] << endl;
	 }
}
