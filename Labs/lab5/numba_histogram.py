import numba
import timeit
from numba import cuda
import cv2
import numpy as np
import time

img = cv2.imread("./image256.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def naive_hist(res, image):
#res[i] = image[image==i].shape[0]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            res[int(image[i][j])] += 1
result_cpu = np.zeros(256)

start = time.perf_counter()
naive_hist(result_cpu, img)
end = time.perf_counter()

print(f"time for cpu implementation: {(end-start)*1000} ms")

@cuda.jit
def hist(result, val):
    tx = cuda.threadIdx.x + cuda.blockDim.x*cuda.blockIdx.x    
    cuda.atomic.add(result, val[tx], 1)

threadsperblock = (128, 1)
blockspergrid = (512,1)
im = img.reshape(-1,)

# calculating average time for 5 runs:
for i in range(5):
    l = []
    result_numba = np.zeros(256)
    start = time.perf_counter()
    hist[blockspergrid,threadsperblock](result_numba, im)
    end = time.perf_counter()
    l.append(end-start)
print(f"time for numba implementation: {sum(l)*200} ms")

if ((result_cpu - result_numba) == np.zeros(256)).all():
    print("The resuls are matching")
else:
    print("The resuls are not matching")

