# Lab 3

Comparing the time for calculating the transpose of square matrices of size 100,
1000, 10000 on cpu, gpu and using shared memory on gpu.

**Kernel timings:**

|       | CPU   | GPU   | GPU (shared memory) |
| :---: | :--:  | :--:  | :------:            |
| 100   | 0.132 | 0.025 | 0.0132              |
| 1000  | 6     | 0.047 | 0.028               |
| 10000 | 1060  | 3.496 | 1.452               |

*\* Time is in milliseconds.*


**Overall timings (includes memory read/writes):**

|       | CPU   | GPU     | GPU (shared memory) |
| :---: | :--:  | :--:    | :------:            |
| 100   | 0.132 | 119.951 | 119.945             |
| 1000  | 7.785 | 182.409 | 182.395             |
| 10000 | 1075  | 872.469 | 870.476             |

*\* Time is in milliseconds.*
