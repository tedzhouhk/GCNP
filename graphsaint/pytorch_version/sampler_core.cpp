#include <stdlib.h>
#include <omp.h>
#include "sampler_core.h"

SamplerCore::SamplerCore(std::vector<int> &adj_indptr_in, std::vector<int> &adj_indices_in, int num_neighbor_in, int num_thread_in = 40)
{
    adj_indptr = adj_indptr_in;
    adj_indices = adj_indices_in;
    num_neighbor = num_neighbor_in;
    num_thread = num_thread_in;
    omp_set_num_threads(num_thread);
}

std::vector<int> SamplerCore::dense_sampling(std::vector<int> &nodes)
{
    std::vector<int> sampled_nodes;
    sampled_nodes.insert(sampled_nodes.begin(), nodes.begin(), nodes.end());
    sampled_nodes.insert(sampled_nodes.end(), nodes.size() * num_neighbor, 0);
#pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < nodes.size(); i++)
        {
            int node = nodes[i];
            for (int j = 0; j < num_neighbor; j++)
            {
                int pos = nodes.size() + i * num_neighbor + j;
                int choose;
                if (adj_indptr[node + 1] - adj_indptr[node] > 0)
                {
                    choose = rand_r(&myseed) % (adj_indptr[node + 1] - adj_indptr[node]);
                    choose = adj_indices[adj_indptr[node] + choose];
                }
                else
                {
                    choose = node;
                }
                sampled_nodes[pos] = choose;
            }
        }
    }
    return sampled_nodes;
}

NodesAdj SamplerCore::sparse_sampling(std::vector<int> &nodes)
{
    std::vector<int> sampled_nodes;
    sampled_nodes.insert(sampled_nodes.begin(), nodes.begin(), nodes.end());
    std::vector<int> sampled_adj_row;
    std::vector<int> sampled_adj_col;
    std::vector<float> sampled_adj_data;
    std::vector<std::vector<int>> sampled_nodes_thread(num_thread);
    std::vector<std::vector<int>> sampled_adj_row_thread(num_thread);
    std::vector<std::vector<int>> sampled_adj_col_thread(num_thread);
    std::vector<std::vector<float>> sampled_adj_data_thread(num_thread);
    std::vector<int> size_thread(num_thread);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < nodes.size(); i++)
        {
            int node = nodes[i];
            int begin = adj_indptr[node];
            int end = adj_indptr[node + 1];
            int deg = end - begin;
            for (int j = begin; j < end; j++)
            {
                sampled_adj_col_thread[tid].push_back(sampled_nodes_thread[tid].size());
                sampled_nodes_thread[tid].push_back(adj_indices[j]);
                sampled_adj_row_thread[tid].push_back(i);
                sampled_adj_data_thread[tid].push_back(static_cast<double>(1) / deg);
            }
        }
        size_thread[tid] = sampled_nodes_thread[tid].size();
    }
    for (auto it = size_thread.begin(); it != size_thread.end() - 1; it++)
    {
        *(it + 1) += *it;
    }
    sampled_nodes.insert(sampled_nodes.end(), size_thread.back(), 0);
    sampled_adj_row.insert(sampled_adj_row.end(), size_thread.back(), 0);
    sampled_adj_col.insert(sampled_adj_col.end(), size_thread.back(), 0);
    sampled_adj_data.insert(sampled_adj_data.end(), size_thread.back(), 0);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int offset = (tid == 0) ? 0 : size_thread[tid - 1];
        int offset_nodes = offset + sampled_nodes.size() - size_thread.back();
        for (int i = 0; i < sampled_nodes_thread[tid].size(); i++)
        {
            sampled_nodes[offset_nodes + i] = sampled_nodes_thread[tid][i];
            sampled_adj_row[offset + i] = sampled_adj_row_thread[tid][i];
            sampled_adj_col[offset + i] = sampled_adj_col_thread[tid][i] + offset_nodes;
            sampled_adj_data[offset + i] = sampled_adj_data_thread[tid][i];
        }
    }
    // for (auto it = nodes.begin(); it != nodes.end(); it++)
    // {
    //     int node = *it;
    //     int begin = adj_indptr[node];
    //     int end = adj_indptr[node + 1];
    //     int deg = end - begin;
    //     for (auto itt = adj_indices.begin() + begin; itt != adj_indices.begin() + end; itt++)
    //     {
    //         int choose = *itt;
    //         sampled_adj_col.push_back(sampled_nodes.size());
    //         sampled_nodes.push_back(choose);
    //         sampled_adj_row.push_back(it - nodes.begin());
    //         sampled_adj_data.push_back(static_cast<double>(1) / deg);
    //     }
    // }
    NodesAdj ans;
    ans.nodes = sampled_nodes;
    ans.adj_row = sampled_adj_row;
    ans.adj_col = sampled_adj_col;
    ans.adj_data = sampled_adj_data;
    return ans;
}