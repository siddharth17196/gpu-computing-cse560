#include <iostream>
#include <math.h>
#include <float.h>


__global__ void sdt_compute(unsigned char *img, int *sz, float *sdt, int sz_edge, int width, float *d_min, int start, int val)
{
  int tx = threadIdx.x + blockDim.x*blockIdx.x;
	extern __shared__ int ep[];
  for(int i=start, j=0;i< val; i++){
    ep[j++] = sz[i];
  }
  __syncthreads();
  float min_dist, dist2;
  min_dist = d_min[tx];
  float _x, _y;
  float sign;
  float dx, dy;
  int x = tx % width;
  int y = tx / width;
  for(int k=0; k<val-start; k++)
  {
    _x = ep[k] % width;
    _y = ep[k] / width;
    dx = _x - x;
    dy = _y - y;
    dist2 = dx*dx + dy*dy;
    if(dist2 < min_dist) min_dist = dist2;
  }
  d_min[tx] = min_dist;

}


__global__ void final_comp(unsigned char *img, float *d_min, float *d_min2, float *sdt)
{
  float sign;
  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  sign  = (img[tx] >= 127)? 1.0f : -1.0f;
  float dm = d_min[tx];
  if(dm > d_min2[tx])
    dm = d_min2[tx];
  sdt[tx] = sign * sqrtf(dm);  
}

extern "C" void run_sampleKernel(unsigned char * bitmap, float *sdt, int width, int height)
{
  //Collect all edge pixels in an array
  int sz = width*height;
  int sz_edge = 0;
  for(int i = 0; i<sz; i++) if(bitmap[i] == 255) sz_edge++;
  int *edge_pixels = new int[sz_edge];
  for(int i = 0, j = 0; i<sz; i++) if(bitmap[i] == 255) edge_pixels[j++] = i;
  std::cout<< "\t"<<sz_edge << " edge pixels in the image of size " << width << " x " << height << "\n"<<std::flush;

  int *d_sz;
  float *temp_min;
  unsigned char *d_img;
  float *d_sdt, *d_min, *d_min2;

  cudaMallocHost(&temp_min,height*width*sizeof(float));
  // temp_min = new float[height*width];

  for(int i=0;i<height*width;i++){
    temp_min[i] = FLT_MAX;
  }

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  // cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);

  cudaMalloc((void**)&d_sz, sz_edge*sizeof(int));
  cudaMalloc((void**)&d_img, height*width*sizeof(unsigned char));
  cudaMalloc((void**)&d_sdt, height*width*sizeof(float));
  cudaMalloc((void**)&d_min, height*width*sizeof(float));
  cudaMalloc((void**)&d_min2, height*width*sizeof(float));

  cudaMemcpyAsync(d_img, bitmap, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_min, temp_min, width*height*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_min2, temp_min, width*height*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_sz, edge_pixels, sz_edge*sizeof(int), cudaMemcpyHostToDevice);

  int divisions = 20;
  int val_div = sz_edge/divisions;
  int n, m;
  m =1;
  for(n =0; n<divisions; n+=2){
    sdt_compute<<<(height*width)/256, 256, val_div*sizeof(int), stream2>>>(d_img, d_sz, d_sdt, sz_edge, width, d_min, n*val_div, (n+1)*val_div); 
    sdt_compute<<<(height*width)/256, 256, val_div*sizeof(int), stream1>>>(d_img, d_sz, d_sdt, sz_edge, width, d_min2, m*val_div, (m+1)*val_div);
    m += 2; 
  }
  // std::cout<<n<<std::endl;
  if((sz_edge%divisions) !=0){
    sdt_compute<<<(height*width)/256, 256, (sz_edge%divisions)*sizeof(int)>>>(d_img, d_sz, d_sdt, sz_edge, width, d_min, (m-1)*val_div, (m-1)*val_div + sz_edge%divisions);
  }

  final_comp<<<(height*width)/256, 256>>>(d_img, d_min, d_min2, d_sdt);
  cudaDeviceSynchronize();
  cudaMemcpy(sdt, d_sdt, height*width*sizeof(float), cudaMemcpyDeviceToHost);

}
