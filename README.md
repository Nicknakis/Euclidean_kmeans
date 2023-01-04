# Euclidean_kmeans 

Python 3.8.3 and Pytorch 1.12.1 implementation of divisive clustering based on Lloyds algorithm and the the Euclidean norm.

## Description

Existing k-means clustering algorithms rely on the squared Euclidean norm distances, and thus Euclidean norms are unfortunately not accounted for. We therefore presently derive an optimization procedure for k-means clustering with the Euclidean norm utilizing an auxiliary function framework yielding closed-form updates until convergence.

Such a divisive clustering procedure relies on the following Euclidean norm objective:

<img src="https://github.com/Nicknakis/Euclidean_kmeans/blob/images/Tex2Img_1672870354.jpg?raw=true" /> 

where $k$ denotes the cluster id, $z_{i}$ is the i'th data observation, $r_{ik}$ the cluster responsibility/assignment, and $\mu_k$ the cluster centroid.




We define an auxiliary function as:

<img src="https://github.com/Nicknakis/Euclidean_kmeans/blob/images/Tex2Img_1672869353.jpg?raw=true" /> 


where $\phi$ are the auxiliary variables. 
Thereby, minimizing Equation $J^+$ with respect to $\phi_{nk}$ yields:

$\phi_{ik}^*=||\mathbf{z}_i-\mu_k||_2$ 

and by plugging $\phi_{ik}^*$ back to $J^+$ we obtain:

$J^+(\phi^*,\mathbf{r},\mu)=J(\mathbf{r},\mathbf{\mu})$ 

verifying that  $J^+$ is indeed a valid auxiliary function for  $J$. 

The algorithm proceeds by optimizing cluster centroids as:

<img src="https://github.com/Nicknakis/Euclidean_kmeans/blob/images/Tex2Img_1672870551.jpg?raw=true" /> 


and assigning points to centroids as:

<img src="https://github.com/Nicknakis/Euclidean_kmeans/blob/images/Tex2Img_1672870449.jpg?raw=true" /> 

upon which $\phi_k$ is updated. The overall complexity of this procedure is $\mathcal{O}(TKND)$ where $T$ is the number of iterations required to converge.

### References
N. Nakis, A. Celikkanat, S. Lehmann and M. Mørup, [A Hierarchical Block Distance Model for Ultra Low-Dimensional Graph Representations](https://arxiv.org/abs/2204.05885), Preprint.

## Installation

### Create a Python 3.8.3 environment with conda

```
conda create -n ${env_name} python=3.8.3  
```

### Activate the environment

```
conda activate ${env_name} 
```

### Please install the required packages

```
pip install -r requirements.txt
```

### Additional packages

Our Pytorch implementation uses the [pytorch_sparse](https://github.com/rusty1s/pytorch_sparse) package. Installation guidelines can be found at the corresponding [Github repository](https://github.com/rusty1s/pytorch_sparse).

#### For a cpu installation please use: 

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html```

#### For a gpu installation please use:

```pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html```

where ${CUDA} should be replaced by either cu102, cu113, or cu116 depending on your PyTorch installation.
