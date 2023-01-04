



import torch
import torch.nn as nn
from torch_sparse import spspmm
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Euclidean_Kmeans():
    def __init__(self,cond_control,k_centers,dimensions,init_cent=None,split_mask=None,previous_cl_idx=None,full_prev_cl=None,prev_centers=None,full_prev_centers=None,centroids_split=None,assigned_points=None,aux_distance=None,local_idx=None,initialization=1,retain_structure=False,CUDA=True,device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), n_iter=300):
        """
        Kmeans-Euclidean Distance minimization: Pytorch CUDA version
        k_centers: number of the starting centroids
        Data:Data array (if already in CUDA no need for futher transform)
        N: Dataset size
        Dim: Dataset dimensionality
        n_iter: max epoch iterations, if the centeroids not shifting any more, the calculation will cease before this max number
        full_cuda_load: send the whole matrix into CUDA for faster implementations (memory requirements)
        
        AVOID the use of dataloader module of Pytorch---every batch will be loaded from the CPU to GPU giving high order loading overhead
        """
        if CUDA:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.k_centers=k_centers
        self.N=dimensions[0]
        self.Dim=dimensions[-1]
        self.device=device
        self.CUDA=CUDA
        self.flag1=0
        self.previous_cl_idx=previous_cl_idx
        self.initialization=initialization
        self.cluster_idx=previous_cl_idx
        self.aux_distance=aux_distance
        #self.splitting_criterion=splitting_criterion
        #if not self.split_flag:
        self.pdist_tol=nn.PairwiseDistance(p=2,eps=0)
        self.collapse_flag=False
        self.cond_control=cond_control
        

            
        if CUDA:
            
            if self.initialization:
                self.lambdas_full = torch.rand(self.N,self.k_centers,device=device)
            else:
                self.lambdas_full = torch.rand(self.N,2,device=device)
            self.local_cl_idx=torch.cuda.LongTensor(self.N).fill_(0)

            self.inv_lambdas_full=1/self.lambdas_full
            if self.initialization:
                self.centroids=init_cent
                self.cluster_idx= torch.cuda.LongTensor(self.N).fill_(0)
            else:
                if retain_structure:
                    self.centroids=prev_centers
                    
                else:
                    self.centroids=torch.cuda.FloatTensor(self.k_centers,self.Dim).fill_(0)
                    even_idx=torch.arange(0, self.k_centers,2)
                    odd_idx=torch.arange(1, self.k_centers,2)
           
                    self.centroids[odd_idx,:]=prev_centers+0.01
                    self.centroids[even_idx,:]=prev_centers-0.01
                collapse_control_avg_radius = torch.cuda.FloatTensor(full_prev_centers.shape[0]).fill_(0)
                
                collapse_control_avg_radius=collapse_control_avg_radius.index_add(0, full_prev_cl, 0.5*self.aux_distance[torch.arange(self.aux_distance.shape[0],device=self.device),local_idx])
                collapse_control_avg_radius=(collapse_control_avg_radius/assigned_points)[centroids_split]
               
                #self.condensed_centers=torch.where(collapse_control_avg_radius<10*(self.Dim**0.5)*self.cond_control)[0]
                self.condensed_centers=(collapse_control_avg_radius<10*(self.Dim**0.5)*self.cond_control).float()
               
                if self.condensed_centers.sum()>0:
                    self.collapse_flag=True
                    # self.collapses=torch.where(self.previous_cl_idx.unsqueeze(-1)==self.condensed_centers)
                    # self.collapsed_nodes=self.collapses[0]
                    # self.collapsed_cnts=self.condensed_centers[self.collapses[1]]
# =============================================================================
                    indicator=torch.cat([torch.arange(centroids_split.sum().long()).unsqueeze(0),torch.zeros(centroids_split.sum().long()).long().unsqueeze(0)],0)
               
                                
                    centers_indicator=torch.cat([torch.arange(self.previous_cl_idx.shape[0]).unsqueeze(0),self.previous_cl_idx.unsqueeze(0)],0)
                    
                    indexC, valueC = spspmm(centers_indicator,torch.ones(centers_indicator.shape[1]), indicator,self.condensed_centers,self.previous_cl_idx.shape[0],centroids_split.shape[0],1,coalesced=True)
                    self.collapsed_nodes=indexC[0][valueC.bool()]
                    self.collapsed_cnts=self.previous_cl_idx[valueC.bool()]
#=============================================================================
               
        else:
            self.lambdas = torch.rand(self.N,self.k_centers)
           
            if self.initialization:
                self.centroids=torch.randn(self.k_centers,self.Dim)

            else:
                self.centroids=torch.randn(int(self.k_centers/2),2,self.Dim)
                self.centroids_binary_extension=self.centroids[self.previous_cl_idx,:,:]
            self.inv_lambdas=torch.zeros(self.N)




        self.n_iter=n_iter
       
    
    def Kmeans_run(self,Data,Data_grad=None):
        '''
        '''
        #self.kmeans_plus_plus()
        for t in range(300):
            if t==0:
                self.Kmeans_step(Data)
                self.previous_centers=self.centroids
            else:
                self.Kmeans_step(Data)
                if self.pdist_tol(self.previous_centers,self.centroids).sum()<1e-4:
                    break
                self.previous_centers=self.centroids
        if self.collapse_flag:
            self.cluster_idx[self.collapsed_nodes]=2*self.collapsed_cnts+torch.randint(0,2,(self.collapsed_cnts.shape[0],))
        if Data_grad==None:
            self.update_clusters(self.cluster_idx, self.sq_distance,Data)
        else:
            self.update_clusters(self.cluster_idx, self.sq_distance,Data_grad)

        #print('total number of iterations:',t)
        #create Z^T responsibility sparse matrix mask KxN 
        if self.initialization:
           
            sparse_mask=torch.sparse.FloatTensor(torch.cat((self.cluster_idx.unsqueeze(-1),torch.arange(self.N).unsqueeze(-1)),1).t(),torch.ones(self.N),torch.Size([self.k_centers,self.N]))
        else:
            sparse_mask=torch.sparse.FloatTensor(torch.cat((self.cluster_idx.unsqueeze(-1),torch.arange(self.N).unsqueeze(-1)),1).t(),torch.ones(self.N),torch.Size([self.k_centers,self.N]))
          
   

        return sparse_mask,self.cluster_idx,self.local_cl_idx,self.aux_distance
  
        
        
        
    def Kmeans_step(self,Data):
        cluster_idx,sq_distance=self.calc_idx(Data)
        self.update_clusters(cluster_idx,sq_distance,Data)

    def calc_idx(self,Data):
        aux_distance,sq_distance=self.calc_dis(Data)
        _, cluster_idx=torch.min(aux_distance,dim=-1)
        self.local_cl_idx=torch.cuda.LongTensor(self.N).fill_(0)
        if self.initialization:
            # path="C:/Users/nnak/agglomerative/"+'facebook'+"/first_mask"
            # cluster_idx=torch.load(path)

            self.local_cl_idx=cluster_idx
            self.cluster_idx=cluster_idx
            # print(cluster_idx.max())
        else:
            self.local_cl_idx=cluster_idx
            self.cluster_idx=self.local_cl_idx+2*self.previous_cl_idx
           
                
        return self.cluster_idx,sq_distance
    
    def calc_dis(self,Data):
       
  

        with torch.no_grad():
            if self.initialization:
                sq_distance=(((Data.unsqueeze(dim=1)-self.centroids.unsqueeze(dim=0))**2).sum(-1))+1e-06
            else:
                    
                sq_distance=((Data.unsqueeze(dim=1)-self.centroids.view(-1,2,self.Dim)[self.previous_cl_idx,:,:])**2).sum(-1)+1e-06
                
        aux_distance=(sq_distance)*self.inv_lambdas_full+self.lambdas_full

        self.aux_distance=aux_distance
        self.sq_distance=sq_distance
        return aux_distance,sq_distance
    

        
    def update_clusters(self,cluster_idx,sq_distance,Data):
       
        if self.CUDA:
            z = torch.cuda.FloatTensor(self.k_centers, self.Dim).fill_(0)
            o = torch.cuda.FloatTensor(self.k_centers).fill_(0)
        else:
            z = torch.zeros(self.k_centers, self.Dim)
            o = torch.zeros(self.k_centers)
       
        self.lambdas_full=sq_distance**0.5+1e-06
        self.inv_lambdas_full=1/self.lambdas_full
        lambdas=self.lambdas_full[torch.arange(self.N,device=self.device),self.local_cl_idx]
        inv_lambdas=1/lambdas
        self.lambdas_X=torch.zeros(self.N, self.Dim)
        self.lambdas_X=torch.mul(Data,inv_lambdas.unsqueeze(-1))
       
             
        # if not self.initialization:

        #     z=z.index_add(0, cluster_idx, self.lambdas_X)
        #     o=o.index_add(0, cluster_idx, inv_lambdas)
       
        idx=torch.ones(self.N).bool()
        z=z.index_add(0, cluster_idx[idx], self.lambdas_X[idx])
        # print(self.lambdas_X.shape)
        o=o.index_add(0, cluster_idx[idx], inv_lambdas[idx])

        self.centroids=torch.mul(z,(1/(o+1e-06)).unsqueeze(-1))
        
        
        

    
    
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from copy import deepcopy
N=5000
cond_control=0.001
cents=10

X, y = make_blobs(n_samples=np.repeat(int(N/10),10), n_features=2)
#X=0.01*X
lab=y

plt.scatter(X[:, 0], X[:, 1], c=y, s= 30000 / len(X), cmap="tab10")
    #plt.axis([0,1,0,1]) ; plt.tight_layout()
plt.title('True latent variables')
plt.xlabel('z1')
plt.ylabel('z2')
#plt.savefig(f"C:/Users/nnak/Finalizing_LSM/train_masks/gnn_pol//TRUE_Z_d_.png",dpi=300,bbox_inches = 'tight')    
plt.show()


latent_z=torch.from_numpy(X).to(device).float()


number_of_remaining_zetas=latent_z.shape[0]
first_centers=torch.randn((cents,2),device=device)

model=Euclidean_Kmeans(cond_control=cond_control,k_centers=int(cents),dimensions= latent_z.shape,init_cent=first_centers)
sparse_mask,cl_idx,local_idx,aux_distance=model.Kmeans_run(deepcopy(latent_z.detach()),latent_z)



cent=model.centroids.detach().cpu().numpy()        
plt.figure(figsize=(8,8))
X=latent_z.detach().cpu().numpy()   
plt.scatter(X[:,0],X[:,1],s=1,c=y)
# plt.savefig('scatter_x.png',dpi=300)
    
plt.scatter(cent[:,0],cent[:,1],s=20,c='black')

# plt.savefig('scatter_x.png',dpi=300)
plt.show() 




