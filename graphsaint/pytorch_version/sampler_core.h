#include <iostream>
#include <vector>

typedef struct NodesAdj
{
    std::vector<int> nodes;
    std::vector<int> adj_row;
    std::vector<int> adj_col;
    std::vector<float> adj_data;
} NodesAdj;

typedef struct NodesGroup
{
    std::vector<int> simple_nodes;
    std::vector<int> normal_nodes;
} NodesGroup;

class SamplerCore
{
public:
    std::vector<int> adj_indptr;
    std::vector<int> adj_indices;
    int num_neighbor;
    int num_thread;
    SamplerCore(std::vector<int> &adj_indptr_in, std::vector<int> &adj_indices_in, int num_neighbor_in, int num_thread_in);
    std::vector<int> dense_sampling(std::vector<int> &nodes);
    NodesAdj sparse_sampling(std::vector<int> &nodes);
};

// class GroupedSamplerCore
// {
// public:
//     std::vector<int> adj_indptr;
//     std::vector<int> adj_indices;
//     int num_neighbor;
//     int num_thread;
//     SamplerCore(std::vector<int> &adj_indptr_in, std::vector<int> &adj_indices_in, int num_neighbor_in, int num_thread_in);
//     std::vector<int> dense_sampling(std::vector<int> &nodes);
//     NodesAdj sparse_sampling(std::vector<int> &nodes);
// };