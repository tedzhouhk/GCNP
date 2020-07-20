# distutils: language = c++

import numpy as np
cimport numpy as np
# cimport openmp
cimport cython
import scipy.sparse as sp
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp cimport bool

cdef inline void npy2vec_int(np.ndarray[int,ndim=1,mode='c'] nda, vector[int]& vec):
    cdef int size = nda.size
    cdef int* vec_c = &(nda[0])
    vec.assign(vec_c,vec_c+size)

cdef inline void npy2vec_float(np.ndarray[float,ndim=1,mode='c'] nda, vector[float]& vec):
    cdef int size = nda.size
    cdef float* vec_c = &(nda[0])
    vec.assign(vec_c,vec_c+size)

cdef extern from "sampler_core.h":
    ctypedef struct NodesAdj:
        vector[int] nodes
        vector[int] adj_row
        vector[int] adj_col
        vector[float] adj_data

cdef extern from "sampler_core.h":
    ctypedef struct MaskedNodes:
        vector[int] nodes
        vector[int] masks
        vector[float] deg_inv

cdef extern from "sampler_core.h":
    cppclass SamplerCore:
        vector[int] adj_indptr;
        vector[int] adj_indices;
        int num_neighbor;
        SamplerCore(vector[int]&, vector[int]&, int, int)
        vector[int] dense_sampling(vector[int]&)
        NodesAdj sparse_sampling(vector[int]&)
        MaskedNodes masked_dense_sampling(vector[int]&)

cdef extern from "sampler_core.h":
    ctypedef struct ApproxNodesAdj:
        vector[int] known_neighbors
        vector[int] unknown_neighbors
        vector[int] adj_row
        vector[int] adj_col
        vector[float] adj_data

cdef extern from "sampler_core.h":
    cppclass ApproxSamplerCore:
        vector[int] adj_indptr
        vector[int] adj_indices
        vector[int] nodes_known
        int num_neighbor
        int num_thread
        ApproxSamplerCore(vector[int]&, vector[int]&, vector[int]&, int, int, int)
        void update_known_idx(vector[int]&)
        vector[int] dense_sampling(vector[int]&)
        ApproxNodesAdj approx_sparse_sampling(vector[int]&)


cdef class MinibatchSampler:
    cdef SamplerCore* cobj

    def __init__(self,np.ndarray[int,ndim=1,mode='c'] adj_indptr,np.ndarray[int,ndim=1,mode='c'] adj_indices,int num_neighbor,int num_thread=40):
        cdef vector[int] adj_indptr_vec
        cdef vector[int] adj_indices_vec
        npy2vec_int(adj_indptr,adj_indptr_vec)
        npy2vec_int(adj_indices,adj_indices_vec)
        self.cobj=new SamplerCore(adj_indptr_vec,adj_indices_vec,num_neighbor,num_thread)

    def dense_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        sampled_nodes_vec=self.cobj.dense_sampling(nodes_vec)
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper.copy())
        return sampled_nodes

    def masked_dense_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        ans_struct=self.cobj.masked_dense_sampling(nodes_vec)
        sampled_nodes_vec=ans_struct.nodes
        mask_vec=ans_struct.masks
        deg_inv_vec=ans_struct.deg_inv
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper.copy())
        arr_int_helper=<int [:mask_vec.size()]>mask_vec.data()
        mask=np.asarray(arr_int_helper.copy()).astype(np.bool)
        arr_float_helper=<float [:deg_inv_vec.size()]>deg_inv_vec.data()
        deg_inv=np.asarray(arr_float_helper.copy())
        return sampled_nodes,mask,deg_inv
    
    def sparse_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        ans_struct=self.cobj.sparse_sampling(nodes_vec)
        sampled_nodes_vec=ans_struct.nodes;
        sampled_adj_row_vec=ans_struct.adj_row;
        sampled_adj_col_vec=ans_struct.adj_col;
        sampled_adj_data_vec=ans_struct.adj_data;
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_row_vec.size()]>sampled_adj_row_vec.data()
        sampled_adj_row=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_col_vec.size()]>sampled_adj_col_vec.data()
        sampled_adj_col=np.asarray(arr_int_helper).copy()
        arr_float_helper=<float [:sampled_adj_data_vec.size()]>sampled_adj_data_vec.data()
        sampled_adj_data=np.asarray(arr_float_helper).copy()
        sampled_adj=sp.coo_matrix((sampled_adj_data,(sampled_adj_row,sampled_adj_col)),shape=(nodes_vec.size(),sampled_nodes_vec.size()))
        return sampled_nodes,sampled_adj

cdef class ApproxMinibatchSampler:
    cdef ApproxSamplerCore* cobj

    def __init__(self,np.ndarray[int,ndim=1,mode='c'] adj_indptr,np.ndarray[int,ndim=1,mode='c'] adj_indices,np.ndarray[int,ndim=1,mode='c'] known_idx,int num_neighbor,int num_nodes, int num_thread=40):
        cdef vector[int] adj_indptr_vec
        cdef vector[int] adj_indices_vec
        cdef vector[int] known_idx_vec
        npy2vec_int(adj_indptr,adj_indptr_vec)
        npy2vec_int(adj_indices,adj_indices_vec)
        npy2vec_int(known_idx,known_idx_vec)
        self.cobj=new ApproxSamplerCore(adj_indptr_vec,adj_indices_vec,known_idx_vec,num_neighbor,num_nodes,num_thread)

    def update_known_idx(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        self.cobj.update_known_idx(nodes_vec)

    def dense_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        sampled_nodes_vec=self.cobj.dense_sampling(nodes_vec)
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper.copy())
        return sampled_nodes
    
    def approx_sparse_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        ans_struct=self.cobj.approx_sparse_sampling(nodes_vec)
        sampled_known_nodes_vec=ans_struct.known_neighbors;
        sampled_unknown_nodes_vec=ans_struct.unknown_neighbors;
        sampled_adj_row_vec=ans_struct.adj_row;
        sampled_adj_col_vec=ans_struct.adj_col;
        sampled_adj_data_vec=ans_struct.adj_data;
        arr_int_helper=<int [:sampled_known_nodes_vec.size()]>sampled_known_nodes_vec.data()
        sampled_known_nodes=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_unknown_nodes_vec.size()]>sampled_unknown_nodes_vec.data()
        sampled_unknown_nodes=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_row_vec.size()]>sampled_adj_row_vec.data()
        sampled_adj_row=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_col_vec.size()]>sampled_adj_col_vec.data()
        sampled_adj_col=np.asarray(arr_int_helper).copy()
        arr_float_helper=<float [:sampled_adj_data_vec.size()]>sampled_adj_data_vec.data()
        sampled_adj_data=np.asarray(arr_float_helper).copy()
        # print(sampled_adj_row)
        # print(sampled_adj_col)
        # print(nodes_vec.size(),sampled_known_nodes_vec.size()+sampled_unknown_nodes_vec.size())
        sampled_adj=sp.coo_matrix((sampled_adj_data,(sampled_adj_row,sampled_adj_col)),shape=(nodes_vec.size(),sampled_known_nodes_vec.size()+sampled_unknown_nodes_vec.size()))
        return sampled_unknown_nodes,sampled_known_nodes,sampled_adj


'''
# from cython.parallel import prange,parallel
from libc.stdlib cimport rand
from libc.stdio cimport printf
from libcpp.algorithm cimport sort

cdef extern from "stdlib.h":
    int RAND_MAX


cdef class MinibatchSampler:
    cdef int num_neighbor
    cdef vector[int] adj_indptr_vec
    cdef vector[int] adj_indices_vec
    
    def __cinit__(self,np.ndarray[int,ndim=1,mode='c'] adj_indptr,np.ndarray[int,ndim=1,mode='c'] adj_indices,int num_neighbor):
        self.num_neighbor=num_neighbor
        npy2vec_int(adj_indptr,self.adj_indptr_vec)
        npy2vec_int(adj_indices,self.adj_indices_vec)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def dense_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        cdef vector[int] sampled_nodes_vec
        sampled_nodes_vec.insert(sampled_nodes_vec.begin(),nodes_vec.begin(),nodes_vec.end())
        cdef int i=0
        cdef int j=0
        cdef int node
        cdef int choose
        with nogil:
            while i<nodes_vec.size():
                node=nodes_vec[i]
                j=0
                while j<self.num_neighbor:
                    choose=rand()%(self.adj_indptr_vec[node+1]-self.adj_indptr_vec[node])
                    choose=self.adj_indices_vec[self.adj_indptr_vec[node]+choose]
                    sampled_nodes_vec.push_back(choose)
                    j=j+1
                i=i+1
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper.copy())
        return sampled_nodes

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sparse_sampling(self,np.ndarray[int,ndim=1,mode='c'] nodes):
        cdef vector[int] nodes_vec
        npy2vec_int(nodes,nodes_vec)
        cdef vector[int] sampled_nodes_vec
        sampled_nodes_vec.insert(sampled_nodes_vec.begin(),nodes_vec.begin(),nodes_vec.end())
        cdef vector[int] sampled_adj_row_vec
        cdef vector[int] sampled_adj_col_vec
        cdef vector[float] sampled_adj_data_vec
        cdef int i=0
        cdef int j=0
        cdef int node
        cdef int choose
        cdef int deg
        with nogil:
            while i<nodes_vec.size():
                node=nodes_vec[i]
                j=self.adj_indptr_vec[node]
                deg=self.adj_indptr_vec[node+1]-self.adj_indptr_vec[node]
                while j<self.adj_indptr_vec[node+1]:
                    choose=self.adj_indices_vec[j]
                    sampled_adj_col_vec.push_back(sampled_nodes_vec.size())
                    sampled_nodes_vec.push_back(choose)
                    sampled_adj_row_vec.push_back(i)
                    sampled_adj_data_vec.push_back(1/(<float>deg))
                    j=j+1
                i=i+1
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        sampled_nodes=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_row_vec.size()]>sampled_adj_row_vec.data()
        sampled_adj_row=np.asarray(arr_int_helper).copy()
        arr_int_helper=<int [:sampled_adj_col_vec.size()]>sampled_adj_col_vec.data()
        sampled_adj_col=np.asarray(arr_int_helper).copy()
        arr_float_helper=<float [:sampled_adj_data_vec.size()]>sampled_adj_data_vec.data()
        sampled_adj_data=np.asarray(arr_float_helper).copy()
        sampled_adj=sp.coo_matrix((sampled_adj_data,(sampled_adj_row,sampled_adj_col)),shape=(nodes_vec.size(),sampled_nodes_vec.size()))
        return sampled_nodes,sampled_adj
'''