#include <iostream>
#include <stdio.h>
#include <time.h>

#define LENGTH 100
using namespace std;

void matmul(int a[][LENGTH], int b[][LENGTH], int c[][LENGTH])
{
	for(int i=0; i<LENGTH; i++)
    {
        for(int j=0; j<LENGTH; j++)
        {
            c[i][j] = 0;
            for(int k=0; k<LENGTH; k++)
		        c[i][j] += a[i][k] * b[k][j];
        }
	}
}

int main(){

	static int a_vec[LENGTH][LENGTH];
	static int b_vec[LENGTH][LENGTH];
	static int h_c[LENGTH][LENGTH];

	struct timespec begin, end; 

	for(int i=0 ; i< LENGTH; i++)
    {
        for(int j=0;j<LENGTH;j++){
		    a_vec[i][j] = i-j;
		    b_vec[i][j] = i+j;
	    }
    }

	clock_gettime(CLOCK_REALTIME, &begin);
	
	matmul(a_vec, b_vec, h_c);

	clock_gettime(CLOCK_REALTIME, &end);
    long seconds = end.tv_sec - begin.tv_sec;
    long nanoseconds = end.tv_nsec - begin.tv_nsec;
    double elapsed = seconds + nanoseconds*1e-9;


	/* for(int i=0 ; i< LENGTH/100000 ; i++){ */
	/* 	std::cout << h_c[i] << std::endl; */
	/* } */

    printf("Time measured: %.3f seconds.\n", elapsed);
}

