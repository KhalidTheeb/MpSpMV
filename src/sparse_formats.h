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

#include "mem.h"

#define BLOCKSIZEN 512
#define BLOCKYDIMN 6
#define NUMTHREADS 23040

template<typename IndexType>
struct matrix_shape
{
    typedef IndexType index_type;
    IndexType num_rows, num_cols, num_nonzeros;
};

// COOrdinate matrix (aka IJV or Triplet format)
template <typename IndexType, typename ValueType>
struct coo_matrix : public matrix_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType * I;  //row indices
    IndexType * J;  //column indices
    ValueType * V;  //nonzero values
};

/*
 *  Compressed Sparse Row matrix (aka CRS)
 */
template <typename IndexType, typename ValueType>
struct csr_matrix : public matrix_shape<IndexType>
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType * Ap;  //row pointer
    IndexType * Aj;  //column indices
    ValueType * Ax;  //nonzeros
};

////////////////////////////////////////////////////////////////////////////////
//! sparse matrix memory management 
////////////////////////////////////////////////////////////////////////////////
template <typename IndexType, typename ValueType>
void delete_csr_matrix(csr_matrix<IndexType,ValueType>& csr, const memory_location loc){
    delete_array(csr.Ap, loc);  delete_array(csr.Aj, loc);   delete_array(csr.Ax, loc);
}

template <typename IndexType, typename ValueType>
void delete_coo_matrix(coo_matrix<IndexType,ValueType>& coo, const memory_location loc){
    delete_array(coo.I, loc);   delete_array(coo.J, loc);   delete_array(coo.V, loc);
}

////////////////////////////////////////////////////////////////////////////////
//! host functions
////////////////////////////////////////////////////////////////////////////////


template <typename IndexType, typename ValueType>
void delete_host_matrix(coo_matrix<IndexType,ValueType>& coo){ delete_coo_matrix(coo, HOST_MEMORY); }

template <typename IndexType, typename ValueType>
void delete_host_matrix(csr_matrix<IndexType,ValueType>& csr){ delete_csr_matrix(csr, HOST_MEMORY); }


////////////////////////////////////////////////////////////////////////////////
//! device functions
////////////////////////////////////////////////////////////////////////////////

template <typename IndexType, typename ValueType>
void delete_device_matrix(coo_matrix<IndexType,ValueType>& coo){ delete_coo_matrix(coo, DEVICE_MEMORY); }

template <typename IndexType, typename ValueType>
void delete_device_matrix(csr_matrix<IndexType,ValueType>& csr){ delete_csr_matrix(csr, DEVICE_MEMORY); }

////////////////////////////////////////////////////////////////////////////////
//! copy to device
////////////////////////////////////////////////////////////////////////////////


template <typename IndexType, typename ValueType>
csr_matrix<IndexType, ValueType> copy_matrix_to_device(const csr_matrix<IndexType, ValueType>& h_csr)
{
    csr_matrix<IndexType, ValueType> d_csr = h_csr; //copy fields
    d_csr.Ap = copy_array_to_device(h_csr.Ap, h_csr.num_rows + 1);
    d_csr.Aj = copy_array_to_device(h_csr.Aj, h_csr.num_nonzeros);
    d_csr.Ax = copy_array_to_device(h_csr.Ax, h_csr.num_nonzeros);
    return d_csr;
}

template <typename IndexType, typename ValueType>
coo_matrix<IndexType, ValueType> copy_matrix_to_device(const coo_matrix<IndexType, ValueType>& h_coo)
{
    coo_matrix<IndexType, ValueType> d_coo = h_coo; //copy fields
    d_coo.I = copy_array_to_device(h_coo.I, h_coo.num_nonzeros);
    d_coo.J = copy_array_to_device(h_coo.J, h_coo.num_nonzeros);
    d_coo.V = copy_array_to_device(h_coo.V, h_coo.num_nonzeros);
    return d_coo;
}


