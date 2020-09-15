#include <stdlib.h>
#include <omp.h>
#include <stdlib.h>
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

MaskedNodes SamplerCore::masked_dense_sampling(std::vector<int> &nodes)
{
    std::vector<int> sampled_nodes;
    std::vector<int> mask;
    std::vector<float> deg_inv;
    sampled_nodes.insert(sampled_nodes.begin(), nodes.begin(), nodes.end());
    sampled_nodes.insert(sampled_nodes.end(), nodes.size() * num_neighbor, 0);
    mask.insert(mask.end(), nodes.size() * num_neighbor, 1);
    deg_inv.insert(deg_inv.end(), nodes.size(), 0);
#pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
#pragma omp for schedule(static)
        for (int i = 0; i < nodes.size(); i++)
        {
            int node = nodes[i];
            int deg = adj_indptr[node + 1] - adj_indptr[node];
            if (deg > num_neighbor)
            {
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
                deg_inv[i] = static_cast<double>(1) / num_neighbor;
            }
            else
            {
                for (int j = 0; j < deg; j++)
                {
                    int pos = nodes.size() + i * num_neighbor + j;
                    sampled_nodes[pos] = adj_indices[adj_indptr[node] + j];
                }
                deg_inv[i] = static_cast<double>(1) / deg;
                for (int j = deg; j < num_neighbor; j++)
                {
                    int mask_pos = i * num_neighbor + j;
                    mask[mask_pos] = 0;
                }
            }
        }
    }
    MaskedNodes ans;
    ans.nodes = sampled_nodes;
    ans.masks = mask;
    ans.deg_inv = deg_inv;
    return ans;
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
    NodesAdj ans;
    ans.nodes = sampled_nodes;
    ans.adj_row = sampled_adj_row;
    ans.adj_col = sampled_adj_col;
    ans.adj_data = sampled_adj_data;
    return ans;
}

ApproxSamplerCore::ApproxSamplerCore(std::vector<int> &adj_indptr_in, std::vector<int> &adj_indices_in, std::vector<int> &nodes_known_in, int num_neighbor_in, int num_nodes, int num_thread_in)
{
    adj_indptr = adj_indptr_in;
    adj_indices = adj_indices_in;
    num_neighbor = num_neighbor_in;
    num_thread = num_thread_in;
    omp_set_num_threads(num_thread);
    nodes_known.insert(nodes_known.begin(), num_nodes, 0);
    for (auto it = nodes_known_in.begin(); it != nodes_known_in.end(); it++)
    {
        nodes_known[*it] = 1;
    }
}

void ApproxSamplerCore::update_known_idx(std::vector<int> &nodes)
{
#pragma omp parallel for
    for (int i = 0; i < nodes.size(); i++)
    {
        nodes_known[nodes[i]] = 1;
    }
    return;
}

std::vector<int> ApproxSamplerCore::dense_sampling(std::vector<int> &nodes)
{
    std::vector<int> sampled_nodes;
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
                int pos = i * num_neighbor + j;
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

ApproxNodesAdj ApproxSamplerCore::approx_sparse_sampling(std::vector<int> &nodes)
{
    std::vector<int> sampled_known_nodes;
    std::vector<int> sampled_unknown_nodes;
    std::vector<int> sampled_adj_row;
    std::vector<int> sampled_adj_col;
    std::vector<float> sampled_adj_data;
    std::vector<std::vector<int>> sampled_known_nodes_thread(num_thread);
    std::vector<std::vector<int>> sampled_unknown_nodes_thread(num_thread);
    std::vector<std::vector<int>> sampled_known_adj_row_thread(num_thread);
    std::vector<std::vector<int>> sampled_unknown_adj_row_thread(num_thread);
    std::vector<std::vector<int>> sampled_known_adj_col_thread(num_thread);
    std::vector<std::vector<int>> sampled_unknown_adj_col_thread(num_thread);
    std::vector<std::vector<float>> sampled_known_adj_data_thread(num_thread);
    std::vector<std::vector<float>> sampled_unknown_adj_data_thread(num_thread);
    std::vector<int> size_known_thread(num_thread);
    std::vector<int> size_unknown_thread(num_thread);
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
                int neigh = adj_indices[j];
                if (nodes_known[neigh] == 0)
                {
                    sampled_unknown_adj_col_thread[tid].push_back(sampled_unknown_nodes_thread[tid].size());
                    sampled_unknown_nodes_thread[tid].push_back(neigh);
                    sampled_unknown_adj_row_thread[tid].push_back(i);
                    sampled_unknown_adj_data_thread[tid].push_back(static_cast<double>(1) / deg);
                }
                else
                {
                    sampled_known_adj_col_thread[tid].push_back(sampled_known_nodes_thread[tid].size());
                    sampled_known_nodes_thread[tid].push_back(neigh);
                    sampled_known_adj_row_thread[tid].push_back(i);
                    sampled_known_adj_data_thread[tid].push_back(static_cast<double>(1) / deg);
                }
            }
        }
        size_known_thread[tid] = sampled_known_nodes_thread[tid].size();
        size_unknown_thread[tid] = sampled_unknown_nodes_thread[tid].size();
    }
    for (int i = 0; i < size_known_thread.size() - 1; i++)
    {
        size_known_thread[i + 1] += size_known_thread[i];
        size_unknown_thread[i + 1] += size_unknown_thread[i];
    }
    sampled_known_nodes.insert(sampled_known_nodes.end(), size_known_thread.back(), 0);
    sampled_unknown_nodes.insert(sampled_unknown_nodes.end(), size_unknown_thread.back(), 0);
    sampled_adj_row.insert(sampled_adj_row.end(), size_known_thread.back() + size_unknown_thread.back(), 0);
    sampled_adj_col.insert(sampled_adj_col.end(), size_known_thread.back() + size_unknown_thread.back(), 0);
    sampled_adj_data.insert(sampled_adj_data.end(), size_known_thread.back() + size_unknown_thread.back(), 0);
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int unknown_offset = (tid == 0) ? 0 : size_unknown_thread[tid - 1];
        int known_offset = (tid == 0) ? 0 : size_known_thread[tid - 1];
        known_offset += size_unknown_thread.back();
        for (int i = 0; i < sampled_unknown_nodes_thread[tid].size(); i++)
        {
            sampled_unknown_nodes[unknown_offset + i] = sampled_unknown_nodes_thread[tid][i];
            sampled_adj_row[unknown_offset + i] = sampled_unknown_adj_row_thread[tid][i];
            sampled_adj_col[unknown_offset + i] = sampled_unknown_adj_col_thread[tid][i] + unknown_offset;
            sampled_adj_data[unknown_offset + i] = sampled_unknown_adj_data_thread[tid][i];
        }
        for (int i = 0; i < sampled_known_nodes_thread[tid].size(); i++)
        {
            sampled_known_nodes[known_offset - size_unknown_thread.back() + i] = sampled_known_nodes_thread[tid][i];
            sampled_adj_row[known_offset + i] = sampled_known_adj_row_thread[tid][i];
            sampled_adj_col[known_offset + i] = sampled_known_adj_col_thread[tid][i] + known_offset;
            sampled_adj_data[known_offset + i] = sampled_known_adj_data_thread[tid][i];
        }
    }
    ApproxNodesAdj ans;
    ans.known_neighbors = sampled_known_nodes;
    ans.unknown_neighbors = sampled_unknown_nodes;
    ans.adj_row = sampled_adj_row;
    ans.adj_col = sampled_adj_col;
    ans.adj_data = sampled_adj_data;
    return ans;
}
