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
#pragma once

///////////////////////////////////
//! CPU SpMV kernels
///////////////////////////////////
#include "sparse_formats.h"

template <typename IndexType, typename ValueType>
void __spmv_csr_serial_host(const IndexType num_rows, 
                            const IndexType * Ap, 
                            const IndexType * Aj, 
                            const ValueType * Ax, 
                            const ValueType * x,    
                                  ValueType * y)    
{
    for (IndexType i = 0; i < num_rows; i++){
        const IndexType row_start = Ap[i];
        const IndexType row_end   = Ap[i+1];
        ValueType sum = y[i];
        for (IndexType jj = row_start; jj < row_end; jj++) {            
            const IndexType j = Aj[jj];  //column index
            sum += x[j] * Ax[jj];
        }
        y[i] = sum; 
    }
}

template <typename IndexType, typename ValueType>
void spmv_csr_serial_host(const csr_matrix<IndexType, ValueType>& csr, 
                          const ValueType * x,  
                                ValueType * y)
{
    __spmv_csr_serial_host(csr.num_rows, csr.Ap, csr.Aj, csr.Ax, x, y);
}


template <typename IndexType, typename ValueType>
void __spmv_coo_serial_host(const IndexType num_nonzeros,
                            const IndexType * rows, 
                            const IndexType * cols, 
                            const ValueType * data, 
                            const ValueType * x,  
                                  ValueType * y)
{
    for (IndexType i = 0; i < num_nonzeros; i++){   
        y[rows[i]] += data[i] * x[cols[i]];
    }
}

template <typename IndexType, typename ValueType>
void spmv_coo_serial_host(const coo_matrix<IndexType, ValueType>& coo, 
                          const ValueType * x,  
                                ValueType * y)
{
    __spmv_coo_serial_host(coo.num_nonzeros, coo.I, coo.J, coo.V, x, y);
}



