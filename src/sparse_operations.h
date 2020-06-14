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

#include <algorithm>
#include "sparse_formats.h"
#include "mem.h"

template <class IndexType, class ValueType>
void sum_csr_duplicates(const IndexType num_rows,
                        const IndexType num_cols, 
                              IndexType * Ap, 
                              IndexType * Aj, 
                              ValueType * Ax)
{
    IndexType * next = new_host_array<IndexType>(num_cols);
    ValueType * sums = new_host_array<ValueType>(num_cols);

    for(IndexType i = 0; i < num_cols; i++){
        next[i] = (IndexType) -1;
        sums[i] = (ValueType)   0;
    }

    IndexType NNZ = 0;

    IndexType row_start = 0;
    IndexType row_end   = 0;


    for(IndexType i = 0; i < num_rows; i++){
        IndexType head = (IndexType)-2;

        row_start = row_end; //Ap[i] may have been changed
        row_end   = Ap[i+1]; //Ap[i+1] is safe

        for(IndexType jj = row_start; jj < row_end; jj++){
            IndexType j = Aj[jj];

            sums[j] += Ax[jj];
            if(next[j] == (IndexType)-1){
                next[j] = head;                        
                head    = j;
            }
        }


        while(head != (IndexType)-2){
            IndexType curr = head; //current column
            head   = next[curr];

            if(sums[curr] != 0){
                Aj[NNZ] = curr;
                Ax[NNZ] = sums[curr];
                NNZ++;
            }

            next[curr] = (IndexType)-1;
            sums[curr] =  0;
        }
        Ap[i+1] = NNZ;
    }

    delete_host_array(next);
    delete_host_array(sums);
}
template <class IndexType, class ValueType>
void sum_csr_duplicates(csr_matrix<IndexType,ValueType>& A){
    sum_csr_duplicates(A.num_rows, A.num_cols, A.Ap, A.Aj, A.Ax);
    A.num_nonzeros = A.Ap[A.num_rows];
}


