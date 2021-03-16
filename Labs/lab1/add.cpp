#include <iostream>
#include <stdio.h>
#include <time.h>

#define LENGTH 100000000
using namespace std;

void vector_add_cpu(float *a, float *b, float *c){
	for(int i=0 ; i< LENGTH ; i++){
		c[i] = a[i] + b[i];
	
	}
	
}

int main(){

	float *a_vec;
	float *b_vec;


	struct timespec begin, end; 
    


	float *c_vec;
	float *d_a, *d_b, *d_c;
	float *h_c;

	h_c = (float*)malloc(LENGTH*sizeof(float));
	a_vec = (float*)malloc(LENGTH*sizeof(float));
	b_vec = (float*)malloc(LENGTH*sizeof(float));

	for(int i=0 ; i< LENGTH; i++){
		a_vec[i] = i;
		b_vec[i] = i;
	}

	clock_gettime(CLOCK_REALTIME, &begin);
	
	vector_add_cpu(a_vec, b_vec, h_c);

	clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;


	for(int i=0 ; i< LENGTH/100000 ; i++){
		std::cout << h_c[i] << std::endl;
	}



    printf("Time measured: %.3f seconds.\n", elapsed);
}