#include <iostream>
#include <chrono>
#define LENGTH 10000

using namespace std;
using namespace std::chrono; 


void transpose(int A[][LENGTH], int trans[][LENGTH]){
    for(int i=0; i< LENGTH; i++){
        for(int j=0; j<LENGTH; j++){
            trans[i][j] = A[j][i];
        }
    }
}

int main(){

    static int A[LENGTH][LENGTH];
    static int trans[LENGTH][LENGTH];
    for(int i=0; i< LENGTH;i++){
        for(int j=0;j<LENGTH;j++)
            A[i][j] = i-j;
    }
    auto start = high_resolution_clock::now();
    transpose(A, trans);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout<<duration.count()<<std::endl;
    return 0;
}
