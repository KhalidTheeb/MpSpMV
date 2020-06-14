# Data-driven Mixed Precision Sparse Matrix Vector Multiplication for GPUs (MpSpMV)

This project reusues code from Nvidia's open source CUSP Library.

## Compilation Command:
```
make
```

## Execution Command:
```
./Mp_SpMV ./matrix.mtx
```

## Sample Output using CUDA/10.1 on V100 GPU:
```
Reading sparse matrix from file (/cant.mtx): done
Inside nnz =2814549 Outside nnz =1192834  Total nnz =4007383
Spliting time :  37.8858 ms
        benchmarking :   0.1267 ms ( 63.26 GFLOP/s 466.7 GB/s)
Out of 62451 ,SDD0= 0 ,SDD1= 0 ,SDD2= 12 ,SDD3= 93 ,SDD4= 961 ,SDD5= 9600 ,SDD6= 29128 ,SDD7= 19617 ,SDD8= 2486 ,SDDi= 554
```
