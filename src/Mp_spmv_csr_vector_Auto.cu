/*
 *  Copyright NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
 
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 
#include <time.h> 
#include "sparse_matrix.h"
#include "sparse_formats.h"
#include "sparse_io.h"
#include "mem.h"
#include "spmv_host.h"
#include "sparse_operations.h"


__global__ void spmv_GPU(const unsigned int num_rows,const float *AxS,const float *xS,const double *xD,const int *AjS,const int *ApS,
						 double *y,const double *AxD,const int *AjD,const int *ApD);

#define THREADS_PER_BLOCK  32
#define THREADS_PER_VECTOR 32

size_t bytes_per_spmv(const csr_matrix<int,float>& mtxS, const csr_matrix<int,double>& mtxD)
{
    size_t bytes = 0;
    bytes += 2*sizeof(int) * mtxS.num_rows;     // row pointer
    bytes += 1*sizeof(int) * mtxS.num_nonzeros; // column index
	bytes += 1*sizeof(int) * mtxD.num_nonzeros; // column index
    bytes += 2*sizeof(float) * mtxS.num_nonzeros ; // A[i,j] and x[j]
	bytes += 2*sizeof(double) * mtxD.num_nonzeros ; // A[i,j] and x[j]
    bytes += 2*sizeof(double) * mtxD.num_rows;     // y[i] = y[i] + ...
    return bytes;
}  
  



int main(int argc,char **argv)
{
  struct sparse_matrix A;
  struct sparse_matrixS AS;
  struct sparse_matrix AD;
  float *xs;
  double *xd;
  double *b;
  double *b2;
  int Range = 2;
    
  cudaEvent_t start_event, stop_event;
  float cuda_elapsed_time;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);
    
  char * mm_filename = NULL;
  for(int i = 1; i < argc; i++){
    if(argv[i][0] != '-'){
        mm_filename = argv[i];
        break;
    }
  }
  
  coo_matrix<int,double> coo = read_coo_matrix<int,double>(mm_filename);
  coo_matrix<int,float> cooS;
  coo_matrix<int,double> cooD;
  size_t coo1_nnz=0;
  size_t coo2_nnz=0;

  for( int i = 0; i < coo.num_nonzeros; i++ ){
    if(coo.V[i] >= (-1*Range)  && coo.V[i] <= Range )
        {coo1_nnz++;}
	else 
		{coo2_nnz++;}	
	}
	
  cooD.num_cols = coo.num_cols;
  cooD.num_rows = coo.num_rows;		
  cooD.num_nonzeros = coo1_nnz;	
  cooD.I = new_host_array<int>(coo1_nnz);
  cooD.J = new_host_array<int>(coo1_nnz);
  cooD.V = new_host_array<double>(coo1_nnz);
	
  cooS.num_cols = coo.num_cols;
  cooS.num_rows = coo.num_rows;
  cooS.num_nonzeros = coo2_nnz;
  cooS.I = new_host_array<int>(coo2_nnz);
  cooS.J = new_host_array<int>(coo2_nnz);
  cooS.V = new_host_array<float>(coo2_nnz);
		
  printf("Inside nnz =%d Outside nnz =%d  Total nnz =%d \n",coo2_nnz,coo1_nnz,coo.num_nonzeros);
	
  coo1_nnz=0;
  coo2_nnz=0;			
	
	//timing split loop
  cudaEventRecord(start_event, 0);
		
  for(size_t i = 0; i < coo.num_nonzeros; i++)
  {
   	if(coo.V[i] >= (-1*Range)  && coo.V[i] <= Range)
    {
        cooD.I[coo1_nnz] = coo.I[i];
        cooD.J[coo1_nnz] = coo.J[i];
        cooD.V[coo1_nnz] = coo.V[i];
		coo1_nnz++;
	}
	else 
	{
		cooS.I[coo2_nnz] = coo.I[i];
        cooS.J[coo2_nnz] = coo.J[i];
        cooS.V[coo2_nnz] = coo.V[i];
        coo2_nnz++;
	}
  }
	
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
  printf("Spliting time : %8.4f ms \n", cuda_elapsed_time); 

  csr_matrix<int,double> csr = coo_to_csr(coo, false);
  csr_matrix<int,float> csrS = coo_to_csr(cooS, false);
  csr_matrix<int,double> csrD = coo_to_csr(cooD, false);
	
  delete_host_matrix(coo);
  delete_host_matrix(cooS);
  delete_host_matrix(cooD);

  A.nnz = csr.num_nonzeros;
  A.ncols = csr.num_cols;
  A.nrows = csr.num_rows;
  A.cols = csr.Aj;
  A.rows = csr.Ap;
  A.vals = csr.Ax; 
  
  AS.nnz = csrS.num_nonzeros;
  AS.ncols = csrS.num_cols;
  AS.nrows = csrS.num_rows;
  AS.cols = csrS.Aj;
  AS.rows = csrS.Ap;
  AS.vals = csrS.Ax; 
  
  AD.nnz = csrD.num_nonzeros;
  AD.ncols = csrD.num_cols;
  AD.nrows = csrD.num_rows;
  AD.cols = csrD.Aj;
  AD.rows = csrD.Ap;
  AD.vals = csrD.Ax; 
    
  int i;
  xs = ((float *)(malloc(sizeof(float ) * A . ncols)));
  xd = ((double *)(malloc(sizeof(double ) * A . ncols)));
  b = ((double *)(malloc(sizeof(double ) * A . nrows)));
  b2 = ((double *)(malloc(sizeof(double ) * A . nrows)));
  srand(2013);
  for (i = 0; i < A . ncols; i++){
	double tmp =1* rand() / (RAND_MAX + 1.0);
        xs[i] = (float) tmp;//1.0;//  
	xd[i] = tmp;//1.0;//
  }
  for (i = 0; i < A . nrows; i++) {
    b[i] = 0;
    b2[i] = 0;
  }

  
  spmv_csr_serial_host<int,double>(csr, xd, b);

  int *devI4Ptr;
  int *devI3Ptr;
  float *devI2Ptrs;
  double *devI2Ptrd;
  float *devI1Ptr;

  int *devI4DPtr;
  int *devI3DPtr;
  double *devI1DPtr;
  double *devO1DPtr;
    
  cudaMalloc(((void **)(&devI1Ptr)),AS.nnz* sizeof(float ));
  cudaMemcpy(devI1Ptr,AS.vals,AS.nnz* sizeof(float ),cudaMemcpyHostToDevice);
  
  cudaMalloc(((void **)(&devI2Ptrs)),A . nrows* sizeof(float ));
  cudaMemcpy(devI2Ptrs,xs,A . nrows* sizeof(float ),cudaMemcpyHostToDevice);
  cudaMalloc(((void **)(&devI2Ptrd)),A . nrows* sizeof(double ));
  cudaMemcpy(devI2Ptrd,xd,A . nrows* sizeof(double ),cudaMemcpyHostToDevice);
    
  cudaMalloc(((void **)(&devI3Ptr)),AS.nnz* sizeof(int ));
  cudaMemcpy(devI3Ptr,AS.cols,AS.nnz* sizeof(int ),cudaMemcpyHostToDevice);
  cudaMalloc(((void **)(&devI4Ptr)),(A . nrows+1)* sizeof(int ));
  cudaMemcpy(devI4Ptr,AS.rows,(A . nrows+1)* sizeof(int ),cudaMemcpyHostToDevice);
  
  cudaMalloc(((void **)(&devO1DPtr)),A . nrows* sizeof(double ));
  cudaMemcpy(devO1DPtr,b2,A . nrows* sizeof(double),cudaMemcpyHostToDevice);
  cudaMalloc(((void **)(&devI1DPtr)),AD . nnz* sizeof(double));
  cudaMemcpy(devI1DPtr,AD.vals,AD . nnz* sizeof(double),cudaMemcpyHostToDevice);
  cudaMalloc(((void **)(&devI3DPtr)),AD . nnz* sizeof(int ));
  cudaMemcpy(devI3DPtr,AD.cols,AD . nnz* sizeof(int ),cudaMemcpyHostToDevice);
  cudaMalloc(((void **)(&devI4DPtr)),(A . nrows+1)* sizeof(int ));
  cudaMemcpy(devI4DPtr,AD.rows,(A . nrows+1)* sizeof(int ),cudaMemcpyHostToDevice);
  
  const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
  const size_t MAX_BLOCKS  = 2048;//cusp::system::cuda::detail::max_active_blocks
  const size_t NUM_BLOCKS = min(MAX_BLOCKS, (A . nrows + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);
  


  size_t num_iterations=500;
  cudaEventRecord(start_event, 0);
  spmv_GPU<<<NUM_BLOCKS,THREADS_PER_BLOCK,0 >>>(A . nrows,devI1Ptr,devI2Ptrs,devI2Ptrd,devI3Ptr,devI4Ptr,devO1DPtr,devI1DPtr,devI3DPtr,devI4DPtr);

  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);

  const double seconds = 3.0;
  const size_t min_iterations = 1;
  const size_t max_iterations = 1000;
  double estimated_time = cuda_elapsed_time/1000.0;

  if (estimated_time == 0)
   	num_iterations = max_iterations;
  else
   	num_iterations = std::min(max_iterations, std::max(min_iterations, (size_t) (seconds / estimated_time)) ); 


  cudaEventRecord(start_event, 0);
  for (i = 0; i< num_iterations; i++){
    spmv_GPU<<<NUM_BLOCKS,THREADS_PER_BLOCK,0 >>>(A . nrows,devI1Ptr,devI2Ptrs,devI2Ptrd,devI3Ptr,devI4Ptr, devO1DPtr,devI1DPtr,devI3DPtr,devI4DPtr);
  }
  
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&cuda_elapsed_time, start_event, stop_event);
 
  double msec_per_iteration = cuda_elapsed_time/num_iterations;
  double sec_per_iteration = msec_per_iteration / 1000.0;
  double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) csr.num_nonzeros / sec_per_iteration) / 1e9;
  double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_spmv(csrS,csrD) / sec_per_iteration) / 1e9;
  printf("\tbenchmarking : %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 
 
  cudaMemcpy(b2,devO1DPtr,A . nrows* sizeof(double ),cudaMemcpyDeviceToHost);
  
  cudaFree(devI1Ptr);
  cudaFree(devI2Ptrs);
  cudaFree(devI2Ptrd);
  cudaFree(devI3Ptr);
  cudaFree(devI4Ptr);
  cudaFree(devO1DPtr);
  cudaFree(devI1DPtr);
  cudaFree(devI3DPtr);
  cudaFree(devI4DPtr);

  for (i = 0; i < A . nrows; i++) {
    double kor = fabs(b2[i] - b[i]);
	if(kor>0.0001)
      printf("Values don't match at %d, expected %f obtained %f\n",i,b[i],b2[i]);
      break;
  }

  int k0,k1,k2,k3,k4,k5,k6,k7,k8,ki;
  k0=k1=k2=k3=k4=k5=k6=k7=k8=ki=0;
  for (i = 0; i < A . nrows; i++) {
	while((int)b[i]!= 0 )//normalizes decimal numbers Hari S idea
	{
		b2[i]=b2[i]/10.0;
		b[i]=b[i]/10.0;
	}

    double kor = fabs(b2[i] - b[i]);
		 if (kor<=0.0000000001) ki++;
    else if (kor<=0.000000001) k8++;
	else if (kor<=0.00000001) k7++;
	else if (kor<=0.0000001) k6++;
	else if (kor<=0.000001) k5++;
	else if (kor<=0.00001) k4++;
	else if (kor<=0.0001) k3++;
	else if (kor<=0.001) k2++;
	else if (kor<=0.01) k1++;
	else k0++;
  }
// SDD stands for significant decimal digit
    printf("Out of %d ,SDD0= %d ,SDD1= %d ,SDD2= %d ,SDD3= %d ,SDD4= %d ,SDD5= %d ,SDD6= %d ,SDD7= %d ,SDD8= %d ,SDDi= %d\n",A . nrows,k0,k1,k2,k3,k4,k5,k6,k7,k8,ki);	  


  free(xs);
  free(xd);
  free(b);
  free(b2);
  free(A . rows);  free(AS . rows);  free(AD . rows);
  free(A . cols);  free(AS . cols);  free(AD . cols);
  free(A . vals);  free(AS . vals);  free(AD . vals);
  if (i != A . nrows) 
    exit(1);
  return 0;
}


__global__ void spmv_GPU(const unsigned int num_rows,
						 const float *AxS,
						 const float *xS,
						 const double *xD,
						 const int *AjS,
						 const int *ApS,
						 double *y,
						 const double *AxD,
						 const int *AjD,
						 const int *ApD)
{

  const size_t VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / THREADS_PER_VECTOR;
	
  __shared__ volatile double sdata[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];  // padded to avoid reduction conditionals
  __shared__ volatile int ptrsS[VECTORS_PER_BLOCK][2];
  __shared__ volatile int ptrsD[VECTORS_PER_BLOCK][2];

  const int thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
  const int thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
  const int vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
  const int vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
  const int num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

  for(int row = vector_id; row < num_rows; row += num_vectors)
  {
    // use two threads to fetch Ap[row] and Ap[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2){
        ptrsS[vector_lane][thread_lane] = ApS[row + thread_lane];
	    ptrsD[vector_lane][thread_lane] = ApD[row + thread_lane];
	}
    const int row_startS = ptrsS[vector_lane][0]; //same as: row_start = Ap[row];
    const int row_endS   = ptrsS[vector_lane][1]; //same as: row_end   = Ap[row+1];
	const int row_startD = ptrsD[vector_lane][0]; //same as: row_start = Ap[row];
    const int row_endD   = ptrsD[vector_lane][1]; //same as: row_end   = Ap[row+1];

    // initialize local sum
	double sum = 0.0;
    // accumulate local sums
 
//Single precision
    if ( row_endS - row_startS > 32)
    {
      // ensure aligned memory access to Aj and Ax
        int jj = row_startS - (row_startS & (THREADS_PER_VECTOR - 1)) + thread_lane;

      // accumulate local sums
        if(jj >= row_startS && jj < row_endS)
           sum += AxS[jj]* xS[AjS[jj]];
      // accumulate local sums
	    for(jj += THREADS_PER_VECTOR; jj < row_endS; jj += THREADS_PER_VECTOR)
            sum += AxS[jj]* xS[AjS[jj]];
    }
    else
    {
      // accumulate local sums
        for(int jj = row_startS + thread_lane; jj < row_endS; jj += THREADS_PER_VECTOR)
            sum += AxS[jj]* xS[AjS[jj]];
    }

//Double precision
	if ( row_endD - row_startD > 32)
    {
      // ensure aligned memory access to Aj and Ax
        int jj = row_startD - (row_startD & (THREADS_PER_VECTOR - 1)) + thread_lane;
      // accumulate local sums
        if(jj >= row_startD && jj < row_endD)
           sum += AxD[jj]* xD[AjD[jj]];
      // accumulate local sums
		
        for(jj += THREADS_PER_VECTOR; jj < row_endD; jj += THREADS_PER_VECTOR)
            sum += AxD[jj]* xD[AjD[jj]];
    }
    else
    {
      // accumulate local sums
        for(int jj = row_startD + thread_lane; jj < row_endD; jj += THREADS_PER_VECTOR)
            sum += AxD[jj]* xD[AjD[jj]];

    }

  // store local sum in shared memory
    sdata[threadIdx.x] = sum;

    double temp=0;
  // reduce local sums to row sum
    if (THREADS_PER_VECTOR > 16) {temp = sdata[threadIdx.x + 16]; sdata[threadIdx.x] = sum += temp;}
    if (THREADS_PER_VECTOR >  8) {temp = sdata[threadIdx.x +  8]; sdata[threadIdx.x] = sum += temp;}
    if (THREADS_PER_VECTOR >  4) {temp = sdata[threadIdx.x +  4]; sdata[threadIdx.x] = sum += temp;}
    if (THREADS_PER_VECTOR >  2) {temp = sdata[threadIdx.x +  2]; sdata[threadIdx.x] = sum += temp;}
    if (THREADS_PER_VECTOR >  1) {temp = sdata[threadIdx.x +  1]; sdata[threadIdx.x] = sum += temp;}

  // first thread writes the result
    if (threadIdx.x == 0)
       y[row] = sdata[threadIdx.x];
    }
}

