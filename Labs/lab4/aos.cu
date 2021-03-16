#include <iostream>
#include <stdio.h>
#include <time.h>


#define LENGTH 10000
using namespace std;

struct aos{
    int a;
    int b;
    int c;
};

__global__ void vector_add(aos *arr){
	
    int i = threadIdx.x ;
	 if  (i < LENGTH)
	    arr[i].c = arr[i].a + arr[i].b; // read 

}

__host__ void vector_add_cpu(float a[], float b[], float *c){
	for(int i=0 ; i< LENGTH ; i++){
		c[i] = a[i] + b[i];
		// std::cout << c[i] << std::endl;
	}
	
}

int main(){

	aos *h_aos;
	aos *d_aos;

	

    h_aos = new aos [LENGTH];


	for(int i=0 ; i< LENGTH; i++){
		h_aos[i].a = i;
		h_aos[i].b = i;
	}




	cudaMalloc((void**)&d_aos, LENGTH*sizeof(aos));
	

	cudaMemcpy(d_aos, h_aos, LENGTH*sizeof(aos), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	float milliseconds = 0;
	float total_time = 0.0;

	// for(int k=0 ; k< 1000 ; k++){
		// cudaEventRecord(start);

		vector_add<<<LENGTH/128, 128>>>(d_aos);

		cudaDeviceSynchronize();

		// cudaEventRecord(stop);

		
		cudaMemcpy(h_aos, d_aos, LENGTH*sizeof(aos), cudaMemcpyDeviceToHost);
		
		// cudaEventSynchronize(stop);

	
		cudaEventElapsedTime(&milliseconds, start, stop);

		total_time += milliseconds;
	// }
	std::cout  << "Time taken : " << milliseconds <<  " Avg time : "  <<  total_time  <<  std::endl;
	for(int i=0; i<10 ;i++){
	    cout << h_aos[i].c << endl;
	}
}
