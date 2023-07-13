# author: jingxiaolun
# date: 2023.07.11
# description: different cluster algorithms (including k-means, Agglomerative clustering...)

import os
import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import kmeans_plusplus

import time
###################################################### User Defined Function ################################################################################
# kmeans_cluster
class KMeans_Cluster(object):
    def __init__(self, n_clusters, random_state=0):
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_algorithm = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    
    def cluster(self, sub_visual_output):
        ''' args:
        sub_visual_output: shape of [N, C]
        '''
        sub_visual_output_ = sub_visual_output.detach().cpu().numpy()
        sub_kmeans = self.cluster_algorithm.fit(sub_visual_output_)
        labels = torch.from_numpy(sub_kmeans.labels_)
        score = silhouette_score(sub_visual_output_, sub_kmeans.labels_)
        return labels, score

# cluster_iteration
class Cluster_Iteration(object):
    def __init__(self, max_n_clusters):
        super().__init__()
        self.max_n_clusters = max_n_clusters

    def iterate(self, sub_visual_output):
        ''' args:
        sub_visual_output: shape of [N, C]
        '''
        seq_length = sub_visual_output.shape[0]
        labels_list, score_list = [], []
        #for n_clusters in range(2, self.max_n_clusters + 1):
        #    if n_clusters >= seq_length:
        #        break
        #    kmeans_cluster = KMeans_Cluster(n_clusters)
        #    labels, score = kmeans_cluster.cluster(sub_visual_output)
        #    labels_list.append(labels)
        #    score_list.append(score)

        #best_k = np.argmax(score_list) + 2
        #labels_best = labels_list[best_k - 2]
        
        kmeans_cluster = KMeans_Cluster(4)
        labels, score = kmeans_cluster.cluster(sub_visual_output)
        labels_best = labels

        sub_visual_output_avg = 0
        for cluster_index in range(4):
            sub_visual_output_avg += (sub_visual_output[labels_best==cluster_index].mean(dim=0))
        return sub_visual_output_avg

# frame_cluster
class Frame_Cluster(object):
    def __init__(self, max_n_clusters):
        super().__init__()
        ''' args:
        max_n_clusters: shape of [1]
        '''
        self.cluster_iteration = Cluster_Iteration(max_n_clusters)

    def cluster_process(self, visual_output):
        seq_batch = visual_output.shape[0]
        visual_output_list = []
        for i in range(seq_batch):
            sub_visual_output = visual_output[i]
            sub_visual_output_avg = self.cluster_iteration.iterate(sub_visual_output)
            visual_output_list.append(sub_visual_output_avg.unsqueeze(0))
        visual_output_concat = torch.cat(visual_output_list, dim=0)
        return visual_output_concat
           
############################################################ Main Function ###############################################################################
if __name__ == '__main__':
    # step1: define visual_output and video_mask 
    visual_output = torch.randn(128, 12, 512).clamp(-1, 1)
    
    # step2: define relevant params of cluster algorithm and kmeans-cluster
    max_n_clusters = 10
    frame_cluster = Frame_Cluster(max_n_clusters)

    # step3: implement cluster process
    start = time.time()
    visual_output_cluster = frame_cluster.cluster_process(visual_output)
    print(visual_output_cluster.shape)
    print(f'cluster time: {np.round(time.time()-start, 2)}s')
