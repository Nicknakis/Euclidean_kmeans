import torch
from Euclidean_Kmeans_ import Euclidean_Kmeans
from torch_sparse import spspmm
from copy import deepcopy


class Tree_kmeans_recursion():
    def __init__(self,minimum_points,init_layer_split,device):
        """
        Kmeans-Euclidean Distance minimization: Pytorch CUDA version
        
        """
        self.minimum_points=minimum_points
        self.init_layer_split=init_layer_split
        self.thetas=[]
        self.max_layers=3*init_layer_split
        self.cond_control=0.001
        self.general_mask=[]
        self.general_cl_id=[]
        self.device=device
        
    

    def kmeans_tree_recursively(self,depth,latent_z):
        self.general_mask=[]
        self.general_cl_id=[]

        self.leaf_centroids=[]
        self.general_centroids_sub=[]
        
        
        self.missing_center_i=[]
        self.missing_center_j=[]
        self.removed_bias_i=[]
        self.removed_bias_j=[]
        self.split_history=[]
        self.total_K=int(self.init_layer_split)
        self.latent_z=latent_z
        self.input_size=self.latent_z.shape[0]
        self.latent_dim=self.latent_z.shape[1]
        initial_cntrs=torch.randn(self.total_K,self.latent_z.shape[1],device=self.device)
     

        #first iteration
        flag_uneven_split=False
        model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=int(self.init_layer_split),dimensions= self.latent_z.shape,init_cent=initial_cntrs,device=self.device)
        sparse_mask,cl_idx,local_idx,aux_distance=model.fit(deepcopy(self.latent_z.detach()),self.latent_z)
        self.general_mask.append(torch.arange(self.input_size,device=self.device))       

        self.general_cl_id.append(cl_idx)
        self.centroids_layer1=model.centroids
        self.general_centroids_sub.append(model.centroids)


        full_prev_cl=cl_idx
        first_centers=model.centroids.detach()
        #flags shows if all clusters will be partioned later on
        split_flag=1
        global_cl=torch.zeros(self.latent_z.shape[0],device=self.device).long()
        initial_mask=torch.arange(self.latent_z.shape[0],device=self.device)
        init_id=0
        sum_leafs=0
       

        #self.thetas.append(theta_approx)
        for i in range(depth):
            if i==self.max_layers:
                self.cond_control=10*self.cond_control
               
            #datapoints in each of the corresponding clusters
            #Compute points in each cluster
            assigned_points= torch.zeros(sparse_mask.shape[0],device=self.device)
            assigned_points[torch.sparse.sum(sparse_mask,1)._indices()[0]]=torch.sparse.sum(sparse_mask,1)._values()
            #Splitting criterion decides if a cluster has enough points to be binary splitted
            self.splitting_criterion=(assigned_points>self.minimum_points)
            #datapoints required for further splitting
            self.mask_split=torch.sparse.mm(sparse_mask.transpose(0,1),self.splitting_criterion.unsqueeze(-1).float())

            self.mask_split=self.mask_split.squeeze(-1).bool()
            #split_flags shows if all cluster in the same level will be partioned in the next layer of the tree       
            split_flag=self.splitting_criterion.sum()==int(sparse_mask.shape[0])
            #if not all of the clusters have enough points to be partioned
            # save them as leaf nodes and create the mask for the analytical calculations
            if not split_flag:
               
                #erion shows the leaf nodes
                erion=(assigned_points<=self.minimum_points) & (assigned_points>0)
                #if we have leaf nodes in this tree level give them global ids
                if erion.sum()>0:
                    #keep the leaf nodes for visualization
                    with torch.no_grad():
                        self.leaf_centroids.append(model.centroids[erion])
                    #sum_leafs makes sure no duplicates exist in the cluster id
                    sum_leafs=sum_leafs+erion.sum()
                    
                    #global unique cluster ids to be used
                    clu_ids=torch.arange(init_id,sum_leafs,device=self.device).float()
                    #vector of K size assigned with unique cluster for the leaf nodes, zero elsewhere
                    cl_vec=torch.zeros(sparse_mask.shape[0],device=self.device)
                    cl_vec[erion]=clu_ids
                    #initial starting point for next level of ids, so no duplicates exist
                    init_id=sum_leafs
                    # print(global_cl.unique(return_counts=True))
                    # mask_leaf allocates the unique cluster id to the proper nodes, e.g. (for the first row if node 1 belongs to cl: 4) 1 0 0 0 * 4 0 0 0 = 4 
                    mask_leaf=torch.sparse.mm(sparse_mask.transpose(0,1),cl_vec.unsqueeze(-1).float())
                    # gl_idx takes the global node ids for the mask
                    gl_idx=torch.sparse.mm(sparse_mask.transpose(0,1),erion.unsqueeze(-1).float())
                    gl_idx=gl_idx.squeeze(-1).bool()
                    # gl_idx2 keeps the local gl_idx for the case of a sub network split i.e. N_total != N split
                    gl_idx2=gl_idx
                    if gl_idx.shape[0]!=global_cl.shape[0]:
                        #in the case of uneven split the initial mask keeps track of the global node ids
                        gl_idx=initial_mask[gl_idx].long()

                    # give the proper cl ids to the proper nodes
                    global_cl[gl_idx]=mask_leaf.long().squeeze(-1)[gl_idx2]
                    self.K_leafs=sum_leafs
           
           
            centers=model.centroids
            if self.splitting_criterion.sum()==0:
                
                break
            
            #local clusters to be splitted
            splited_cl_ids_i=torch.where(self.splitting_criterion.float()==1)[0]

            if not split_flag:
                flag_uneven_split=True
                # rename ids so they start from 0 to total splitted
                splited_cl_ids_j=torch.arange(splited_cl_ids_i.shape[0],device=self.device)
                # create sparse locations of K_old x K_new
                index_split=torch.cat((splited_cl_ids_i.unsqueeze(0),splited_cl_ids_j.unsqueeze(0)),0)
                # rename old ids K_old to K_new, NxK mm KxK_new (this matrix acts as the translator, i.e the previous first to be splitted i.e. 5 now becomes the zero one)
                self.ind, self.val = spspmm(sparse_mask._indices()[[1,0]],torch.ones(sparse_mask._indices().shape[1],device=self.device),index_split,torch.ones(splited_cl_ids_j.shape[0],device=self.device),sparse_mask.shape[1],self.splitting_criterion.shape[0],splited_cl_ids_i.shape[0],coalesced=True)
                initial_mask=initial_mask[self.ind[0,:].long()]
                self.mask_split=initial_mask
                cl_idx=self.ind[1,:].long()
            if flag_uneven_split:
                #translate the different size of splits to the total N of the network
                self.mask_split=initial_mask
                
            model=Euclidean_Kmeans(cond_control=self.cond_control,k_centers=2*int(splited_cl_ids_i.shape[0]),dimensions= self.latent_z[self.mask_split].shape,split_mask=self.mask_split,previous_cl_idx=cl_idx,full_prev_cl=full_prev_cl,prev_centers=centers.detach()[self.splitting_criterion],full_prev_centers=centers.detach(),centroids_split=self.splitting_criterion,assigned_points=assigned_points,aux_distance=aux_distance,local_idx=local_idx,initialization=0,device=self.device)
            sparse_mask,cl_idx,local_idx,aux_distance=model.fit(deepcopy(self.latent_z.detach()[self.mask_split]),self.latent_z[self.mask_split])
            full_prev_cl=cl_idx
            if cl_idx.shape[0]==self.input_size:
                self.general_mask.append(torch.arange(self.input_size,device=self.device))     
            else:
                self.general_mask.append(self.mask_split)
            starting_center_id=self.total_K
            self.total_K+=2*int(self.splitting_criterion.sum())
            self.general_cl_id.append(cl_idx+starting_center_id)
            self.general_centroids_sub.append(model.centroids)
            

        return first_centers
        
    
        