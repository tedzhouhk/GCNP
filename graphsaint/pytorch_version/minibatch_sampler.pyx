# distutils: language = c++

import numpy as np
cimport numpy as np
# cimport openmp
cimport cython
import scipy.sparse as sp
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

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
    cppclass SamplerCore:
        vector[int] adj_indptr;
        vector[int] adj_indices;
        int num_neighbor;
        SamplerCore(vector[int]&, vector[int]&, int, int)
        vector[int] dense_sampling(vector[int]&)
        NodesAdj sparse_sampling(vector[int]&)

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