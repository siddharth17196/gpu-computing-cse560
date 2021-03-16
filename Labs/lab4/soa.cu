#include <iostream>
#include <stdio.h>
#include <time.h>


#define LENGTH 256
using namespace std;

struct soa{
    int *a;
    int *b;
    int *c;
};

__global__ void vector_add(int *a, int *b, int *c){
	
    int i = threadIdx.x ;
	 if  (i < LENGTH)
	    c[i] = a[i] + b[i]; // read 

}

__host__ void vector_add_cpu(float a[], float b[], float *c){
	for(int i=0 ; i< LENGTH ; i++){
		c[i] = a[i] + b[i];
		// std::cout << c[i] << std::endl;
	}
	
}

int main(){

	soa h_s;
	int *d_s, *d_a, *d_b;
    
    h_s.a = new int [LENGTH];
    h_s.b = new int [LENGTH];
    h_s.c = new int [LENGTH];


	for(int i=0 ; i< LENGTH; i++){
		h_s.a[i] = i;
		h_s.b[i] = i;
	}




	cudaMalloc((void**)&d_s, LENGTH*sizeof(int));
	cudaMalloc((void**)&d_a, LENGTH*sizeof(int));
	cudaMalloc((void**)&d_b, LENGTH*sizeof(int));

	cudaMemcpy(d_a, h_s.a, LENGTH*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_s.b, LENGTH*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_s, h_s.c, LENGTH*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	float milliseconds = 0;
	float total_time = 0.0;

	// for(int k=0 ; k< 1000 ; k++){
		// cudaEventRecord(start);

		vector_add<<<LENGTH/128, 128>>>(d_a, d_b, d_s);

		cudaDeviceSynchronize();

		// cudaEventRecord(stop);

		
		cudaMemcpy(h_s.c, d_s, LENGTH*sizeof(int), cudaMemcpyDeviceToHost);
		
		// cudaEventSynchronize(stop);

	
		cudaEventElapsedTime(&milliseconds, start, stop);

		total_time += milliseconds;
	// }
	std::cout  << "Time taken : " << milliseconds <<  " Avg time : "  <<  total_time / 1000  <<  std::endl;
	for(int i=0; i<10 ;i++){
	    cout << h_s.c[i] << endl;
	}
}
