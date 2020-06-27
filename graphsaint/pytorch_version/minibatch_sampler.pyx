# distutils: language = c++

import numpy as np
cimport numpy as np
# cimport openmp
cimport cython
import scipy.sparse as sp
from libcpp.vector cimport vector
# from cython.parallel import prange,parallel
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp.algorithm cimport sort

cdef extern from "stdlib.h":
    int RAND_MAX

cdef inline void npy2vec_int(np.ndarray[int,ndim=1,mode='c'] nda, vector[int]& vec):
    cdef int size = nda.size
    cdef int* vec_c = &(nda[0])
    vec.assign(vec_c,vec_c+size)

cdef inline void npy2vec_float(np.ndarray[float,ndim=1,mode='c'] nda, vector[float]& vec):
    cdef int size = nda.size
    cdef float* vec_c = &(nda[0])
    vec.assign(vec_c,vec_c+size)

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
        cdef double start_t,end_t
        start_t=openmp.omp_get_wtime()
        cdef vector[int] nodes_vec
        cdef vector[int] sampled_nodes_vec
        cdef vector[int] sampled_adj_row_vec
        cdef vector[int] sampled_adj_col_vec
        cdef vector[float] sampled_adj_data_vec
        npy2vec_int(nodes,nodes_vec)
        cdef int begin=0
        cdef int end=nodes_vec.size()
        sampled_nodes_vec.insert(sampled_nodes_vec.begin(),nodes_vec.begin(),nodes_vec.end())
        cdef int[::1] arr_int_helper
        cdef float[::1] arr_float_helper
        supports=list()
        aggregate_adjs=list()
        arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
        supports.append(np.asarray(arr_int_helper).copy()
        cdef vector[vector[int]] local_row=vector[vector[int]](self.num_thread)
        cdef vector[vector[int]] local_col=vector[vector[int]](self.num_thread)
        cdef vector[vector[float]] local_data=vector[vector[float]](self.num_thread)
        cdef vector[vector[int]] local_nodes=vector[vector[int]](self.num_thread)
        cdef vector[int] offset=vector[int](self.num_thread)
        cdef int i
        cdef int deg
        cdef int choose
        cdef int node_idx
        cdef int node
        cdef int b
        cdef int tid
        cdef int avg_deg
        cdef int bi=0
        with nogil:
            while bi<self.budget_vec.size():
                b=self.budget_vec[bi]
                if b>=0:
                    avg_deg=b
                else:
                    avg_deg=<int>(self.adj_indices_vec.size()/self.adj_indptr_vec.size()*1.5)
                start_t=openmp.omp_get_wtime()
                for tid in prange(self.num_thread,schedule='static',num_threads=self.num_thread):
                    local_row[tid].clear()
                    local_col[tid].clear()
                    local_data[tid].clear()
                    local_nodes[tid].clear()
                    local_row[tid].reserve(<int>((end-begin)*avg_deg/self.num_thread))
                    local_col[tid].reserve(<int>((end-begin)*avg_deg/self.num_thread))
                    local_data[tid].reserve(<int>((end-begin)*avg_deg/self.num_thread))
                    local_nodes[tid].reserve(<int>((end-begin)*avg_deg/self.num_thread))
                for node_idx in prange(begin,end,schedule='static',num_threads=self.num_thread):
                    tid=openmp.omp_get_thread_num()
                    node=sampled_nodes_vec[node_idx]
                    if b>=0:
                        if self.adj_indptr_vec[node+1]-self.adj_indptr_vec[node]>0:
                            i=0
                            while i<b:
                                choose=rand()%(self.adj_indptr_vec[node+1]-self.adj_indptr_vec[node])
                                choose=self.adj_indices_vec[self.adj_indptr_vec[node]+choose]
                                local_col[tid].push_back(local_row[tid].size())
                                local_row[tid].push_back(node_idx)
                                local_data[tid].push_back(1/(<float>b))
                                local_nodes[tid].push_back(choose)
                                i=i+1
                    else:
                        i=self.adj_indptr_vec[node]
                        deg=self.adj_indptr_vec[node+1]-self.adj_indptr_vec[node]
                        while i<self.adj_indptr_vec[node+1]:
                            choose=self.adj_indices_vec[i]
                            local_col[tid].push_back(local_row[tid].size())
                            local_row[tid].push_back(node_idx)
                            local_data[tid].push_back(1/(<float>deg))
                            local_nodes[tid].push_back(choose)
                            i=i+1
                # end_t=openmp.omp_get_wtime()
                # printf("part 1 time: %f\n",end_t-start_t)
                # start_t=openmp.omp_get_wtime()
                i=0
                while i<self.num_thread:
                    offset[i]=sampled_nodes_vec.size()
                    sampled_adj_row_vec.insert(sampled_adj_row_vec.end(),local_row[i].begin(),local_row[i].end())
                    sampled_adj_data_vec.insert(sampled_adj_data_vec.end(),local_data[i].begin(),local_data[i].end())
                    sampled_nodes_vec.insert(sampled_nodes_vec.end(),local_nodes[i].begin(),local_nodes[i].end())
                    i=i+1
                # handle offset for adj_col
                for tid in prange(self.num_thread,schedule='static',num_threads=self.num_thread):
                    i=0
                    while i<local_col[tid].size():
                        local_col[tid][i]=local_col[tid][i]+offset[tid]
                        i=i+1
                i=0
                while i<self.num_thread:
                    sampled_adj_col_vec.insert(sampled_adj_col_vec.end(),local_col[i].begin(),local_col[i].end())
                    i=i+1
                bi=bi+1
                # end_t=openmp.omp_get_wtime()
                # printf("part 2 time: %f\n",end_t-start_t)
                # need to copy as the vector reallocates when capcity is not enough
                with gil:
                    arr_int_helper=<int [:sampled_nodes_vec.size()]>sampled_nodes_vec.data()
                    supports.append(np.asarray(arr_int_helper).copy())
                    arr_int_helper=<int [:sampled_adj_row_vec.size()]>sampled_adj_row_vec.data()
                    sampled_adj_row=np.asarray(arr_int_helper.copy())
                    arr_int_helper=<int [:sampled_adj_col_vec.size()]>sampled_adj_col_vec.data()
                    sampled_adj_col=np.asarray(arr_int_helper.copy())
                    arr_float_helper=<float [:sampled_adj_data_vec.size()]>sampled_adj_data_vec.data()
                    sampled_adj_data=np.asarray(arr_float_helper).copy()
                    begin=end
                    end=sampled_nodes_vec.size()
                    aggregate_adjs.append(sp.coo_matrix((sampled_adj_data,(sampled_adj_row,sampled_adj_col)),shape=(begin,end)))
        end_t=openmp.omp_get_wtime()
        # printf("cpp reported time: %fs\n",end_t-start_t)
        return supports,aggregate_adjs
        '''