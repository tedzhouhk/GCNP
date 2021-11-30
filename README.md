# Accelerating Large Scale Real-Time GNN Inference using Channel Pruning

# Dependencies

* python >= 3.6.8
* pytorch >= 1.1.0
* cython >=0.29.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* scikit-learn >= 0.19.1
* pyyaml >= 3.12
* g++ >= 5.4.0
* openmp >= 4.0

# Dataset

We use the same data format as GraphSAINT, all dataset used could be downloaded on [Google Drive link](https://drive.google.com/open?id=1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) and put into the ``data`` folder in the root directory. For the Arxiv dataset, please use the undirected ``ogbn-arxiv_undirected`` version instead of the directed ``ogbn-arxiv`` version.

# Model Parameters

All the parameters used in training, pruning and re-training are stored in .yml files in */train_config/fullbatch* and */train_config/minibatch*. The *fullbatch* and *minibatch* folders include the parameters on all datasets for full inference and batched inference, respectively. For each dataset and each inference type, we provide three pruning budgets, 2x, 4x and 8x. Each *.yml* file includes five sections: **1. network** specifying GNN architecture **2. params** parameters in the GNN network **3. phase** parameters to train the original model **4. prune** parameters to prune the trained model **5. retrain_phase** parameters to retrain the pruned model and one optional section **6. batch_inference** parameters in batched inference. The first three sections use the same format as GraphSAINT (GraphSAINT: Graph Sampling Based Inductive Learning Method, Zeng et al, 2020). The detailed information on the entries in the other sections are as follows.

- **prune**:
  - **budget**: a list specifying the budget in each layer. As we prune the network reversely from output to input, the order in the list also starts with the output layer. (i.e., for a 2 layer GNN, the list should be \<classifier\>\<GNN layer-2\>\<GNN layer-1\> ).
  - **dynamic**: *manual* or *static*, specifying self and neighbor features ratio in GNN layer-1. If *manual* is select, the budget for self and neighbor feature need to be manually entered in the **manual_neigh** and **manual_self** entries.
  - **total_step**: number of optimization iterations for the two sub-problems. We find that one iteration delivers similar result as multiple iterations so it is set to be 1.
  - **beta_clip**: *true* or *false*. If *true*, then the larger values in $\beta^{(i)}$ would be clipped.
  - **beta_batch**: batch size used in the sub-problem of $\beta^{(i)}$.
  - **beta_lr**: learning rate used in the sub-problem of $\beta^{(i)}$.
  - **beta_epoch**: number of epochs to optimize the sub-problem of $\beta^{(i)}$.
  - **beta_lmbd_1**: a list specifying the initial value of the initial penalty coefficient $\lambda$ in each layer.
  - **beta_lmbd_1_step**: a list specifying the step size of the penalty coefficient $\lambda$ in each layer. At the end of each epoch in the sub-problem of $\beta^{(i)}$, we increase $\lambda$ by this amount.
  - **weight_batch**: batch size used in the sub-problem of $W^{(i)}$.
  - **weight_lr**: learning rate used in the sub-problem of $W^{(i)}$.
  - **weight_epoch**: number of epochs to optimize the sub-problem of $W^{(i)}$.
- **retrain_phase**: parameters to retrain the pruned GNN network. The same format is used as the **phase** section.
- **batch_inference**:
  - **batch_size**: number of nodes in each inference query of batched inference.
  - **neighbors**: upper bound on number of hop-$x$ $(x\geq2)$ neighbors.
  - **approx**: *true* or *false*. If *true*, apply the technique to store and reuse the hidden features of visited nodes.

# Run

We have two Cython modules that need to be compiled before running. To compile the modules, run the following from the root directory:

```
python GNN/setup.py build_ext --inplace
python GNN/pytorch_version/setup.py build_ext --inplace
```

To run the code

```
python -m GNN.pytorch_version.train --data_perfix <path-to-dataset-folder> --train_config <path-to-config-file>
```

We have also set up some useful flags

- **--gpu**: to specify which GPU to use.
- **--cpu_prune**: to solve the two sub-problems with both CPU and GPU (to save GPU memory).
- **--cpu_minibatch**: to perform batched inference on CPU instead of GPU.
- **--profile_fullbatch**: to apply torch.autograd.profiler.profile to full inference and to print the results. *Note: this will incur extra execution time*
- **--profile_minibatch**: to apply torch.autograd.profiler.profile to batched inference without hidden features and to print the results. *Note: this will incur extra execution time*
- **--profile_approxminibatch**: to apply torch.autograd.profiler.profile to batched inference with hidden features and to print the results. *Note: this will incur extra execution time*

# Citation

Here's the bibtex in case you want to cite our work.
```
@article{10.14778/3461535.3461547,
  author = {Zhou, Hongkuan and Srivastava, Ajitesh and Zeng, Hanqing and Kannan, Rajgopal and Prasanna, Viktor},
  title = {Accelerating Large Scale Real-Time GNN Inference Using Channel Pruning},
  year = {2021},
  issue_date = {May 2021},
  publisher = {VLDB Endowment},
  volume = {14},
  number = {9},
  issn = {2150-8097},
  url = {https://doi.org/10.14778/3461535.3461547},
  doi = {10.14778/3461535.3461547},
  journal = {Proc. VLDB Endow.},
  month = {may},
  pages = {1597–1605},
  numpages = {9}
}
```
