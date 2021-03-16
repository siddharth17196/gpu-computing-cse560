# Lab 4

Vector Addition using Structure of Arrays(SoA) and Array of Structures(AoS)
and measuring some metrics to understand memory coalescing.

**Metrics**:
- WLI - Warp level instructions for global loads
- ELS - Executed Load/Store Instructions
- GLE - Global Memory Load Efficiency
- GSE - Global Memory Store Efficiency
- GLT - Global Load Transactions
- GST - Global Store Transactions


| Implementation\Metrics   | WLI   | ELS   | GLE   | GSE   | GLT   | GST   |
| :----------------------: | :---: | :---: | :---: | :---: | :---: | :---: |
| Array of Structures      | 624   | 1560  | 33.3% | 33.3% | 7488  | 3744  |
| Structure of Arrays      | 16    | 40    | 100%  | 100%  | 64    | 32    |


**Commands:**
```cuda
make
nvprof -m all ./aos.out
nvprof -m all ./soa.out
```


**Length of array was 10,000 in all cases.
