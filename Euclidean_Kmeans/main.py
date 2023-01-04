from tree_recursion import Tree_kmeans_recursion     
from sklearn.datasets import make_blobs
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.style.use('seaborn')


# reproducibility
seed=7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# create data
N=5000
cents=5

X, y = make_blobs(n_samples=np.repeat(int(N/cents),cents), n_features=2)
lab=y


#  transfer to torch
latent_z=torch.from_numpy(X).to(device).float()

# parameters

# minimum allowed points for a cluster to contain in order to be divided in the next tree layer
minimum_points=150

# number of clusters for the data to be splitted in the first level of the tree
init_layer_split=cents

# initialize module
tree=Tree_kmeans_recursion(minimum_points=minimum_points,init_layer_split=cents,device=device)

# run divisive Euclidean K-means
tree.kmeans_tree_recursively(depth=80, latent_z=latent_z)



# visualize
 
from scipy.spatial import ConvexHull

             
for j in range(len(tree.general_cl_id)):   
    cl_id=tree.general_cl_id[j].cpu().numpy()
    cent=tree.general_centroids_sub[j].detach().cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(8,8))
    # plot data
    plt.title('Data Scatter Plot')
    plt.scatter(X[:,0],X[:,1],s=5,cmap="tab10")
    # plot centers
    plt.scatter(cent[:,0],cent[:,1],s=50,c='black',marker='^')
    plt.xlabel('X')
    plt.ylabel('Y')
    # draw enclosure
    if cl_id.shape[0]!=N:
        X_=X[tree.general_mask[j].cpu().numpy()]
    else:
        X_=X
    
    for i in np.unique(cl_id):
        points = X_[cl_id==i]
        # get convex hull
        hull = ConvexHull(points)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(points[hull.vertices,0],
                           points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                           points[hull.vertices,1][0])
        # plot shape
        plt.fill(x_hull, y_hull, alpha=0.5)#, c=colors[i])
        
plt.show()
                    