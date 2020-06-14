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
#include "sparse_operations.h"

////////////////////////////////////////////////////////////////////////////////
//! Convert COO format to CSR format
// Storage for output is assumed to have been allocated
//! @param rows           COO row array
//! @param cols           COO column array
//! @param data           COO data array
//! @param num_rows       number of rows
//! @param num_cols       number of columns
//! @param num_nonzeros   number of nonzeros
//! @param Ap             CSR pointer array
//! @param Ai             CSR index array
//! @param Ax             CSR data array
////////////////////////////////////////////////////////////////////////////////
template <class IndexType, class ValueType>
void coo_to_csr(const IndexType * rows,
                const IndexType * cols,
                const ValueType * data,
                const IndexType num_rows, 
                const IndexType num_cols, 
                const IndexType num_nonzeros,
                      IndexType * Ap,
                      IndexType * Aj,
                      ValueType * Ax)
{
    for (IndexType i = 0; i < num_rows; i++)
        Ap[i] = 0;

    for (IndexType i = 0; i < num_nonzeros; i++)
        Ap[rows[i]]++;


    //cumsum the nnz per row to get Bp[]
    for(IndexType i = 0, cumsum = 0; i < num_rows; i++){     
        IndexType temp = Ap[i];
        Ap[i] = cumsum;
        cumsum += temp;
    }
    Ap[num_rows] = num_nonzeros;

    //write Aj,Ax into Bj,Bx
    for(IndexType i = 0; i < num_nonzeros; i++){
        IndexType row  = rows[i];
        IndexType dest = Ap[row];

        Aj[dest] = cols[i];
        Ax[dest] = data[i];

        Ap[row]++;
    }

    for(IndexType i = 0, last = 0; i <= num_rows; i++){
        IndexType temp = Ap[i];
        Ap[i]  = last;
        last   = temp;
    }
    
}


////////////////////////////////////////////////////////////////////////////////
//! Convert COOrdinate format (triplet) to CSR format
//! @param coo        coo_matrix
////////////////////////////////////////////////////////////////////////////////
template <class IndexType, class ValueType>
csr_matrix<IndexType, ValueType>
 coo_to_csr(const coo_matrix<IndexType,ValueType>& coo, bool compact = false){  

    csr_matrix<IndexType, ValueType> csr;

    csr.num_rows     = coo.num_rows;
    csr.num_cols     = coo.num_cols;
    csr.num_nonzeros = coo.num_nonzeros;

    csr.Ap = new_host_array<IndexType>(csr.num_rows + 1);
    csr.Aj = new_host_array<IndexType>(csr.num_nonzeros);
    csr.Ax = new_host_array<ValueType>(csr.num_nonzeros);

    coo_to_csr(coo.I, coo.J, coo.V,
               coo.num_rows, coo.num_cols, coo.num_nonzeros,
               csr.Ap, csr.Aj, csr.Ax);
    
    if (compact) {
        //sum duplicates together
        sum_csr_duplicates(csr.num_rows, csr.num_cols, csr.Ap, csr.Aj, csr.Ax);
        csr.num_nonzeros = csr.Ap[csr.num_rows];
    }

    return csr;
}



