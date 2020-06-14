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
#ifndef __SPARSE_MATRIX_STRUCT__
#define __SPARSE_MATRIX_STRUCT__

#define SIZE 16
#define BLOCKSIZE 4

#define DEBUG 1
#define BILLION 1E9
typedef float REAL;


struct sparse_matrix {
        int max_row_len;
	int nrows;
	int ncols;	
	int nnz;
	int *rows;
	int *cols;
	double *vals;
};


struct sparse_matrixS {
        int max_row_len;
	int nrows;
	int ncols;	
	int nnz;
	int *rows;
	int *cols;
	float *vals;
};

struct sparse_args {
        struct sparse_matrix *s;
        REAL *x;
        REAL *b;
        int nt;
        int id;
};
#endif

