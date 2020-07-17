#include <iostream>
#include <vector>
#include <unordered_set>

typedef struct NodesAdj
{
    std::vector<int> nodes;
    std::vector<int> adj_row;
    std::vector<int> adj_col;
    std::vector<float> adj_data;
} NodesAdj;

typedef struct ApproxNodesAdj
{
    std::vector<int> known_neighbors;
    std::vector<int> unknown_neighbors;
    std::vector<int> adj_row;
    std::vector<int> adj_col;
    std::vector<float> adj_data;
} ApproxNodesAdj;

typedef struct MaskedNodes
{
    std::vector<int> nodes;
    std::vector<int> masks;
    std::vector<float> deg_inv;
} MaskedNodes;

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
    MaskedNodes masked_dense_sampling(std::vector<int> &nodes);
};

class ApproxSamplerCore
{
public:
    std::vector<int> adj_indptr;
    std::vector<int> adj_indices;
    std::unordered_set<int> nodes_known;
    int num_neighbor;
    int num_thread;
    ApproxSamplerCore(std::vector<int> &adj_indptr_in, std::vector<int> &adj_indices_in, std::vector<int> &nodes_known_in, int num_neighbor_in, int num_thread_in);
    void update_known_idx(std::vector<int> &nodes);
    std::vector<int> dense_sampling(std::vector<int> &nodes);
    ApproxNodesAdj approx_sparse_sampling(std::vector<int> &nodes);
};