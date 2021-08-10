import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import data
from skimage import io
from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean
import scipy.ndimage as ndimg
import cv2
import cv2.xfeatures2d as f2d
import random
import os
from main import *

imgdb, features = build_feature_db('./DVDcovers')
INDICES = list(range(features.shape[0]))

class VocabTree:
    def __init__(self, branch_factor, num_levels, k_means, features, _id, _batch_size=8):
        self.bf = branch_factor
        
        self.value = None
        self.children = [None] * self.bf
        self.id = _id
        
        self.L = num_levels
        self.K = branch_factor
        self.features = features
        self.k_means_func = k_means
        self.KMeans = k_means(n_clusters=self.K, random_state=1, batch_size=_batch_size)
        
        self.num_nodes = features.shape[0]
        self.imgdb = {}
        
    def set_parent(self, p):
        self.parent = p
        
    def set_child(self, i, c):
        self.children[i] = c
        
    def set_root_value(self, value):
        self.value = value
        
    def write_tree(self, t):
        self.tree = t
        
    def is_leaf_node(self):
        return all((x == None) for x in self.children)
        
    def get_cluster_centroid(self, cluster):
        S = np.sum(cluster, axis=0)
        min_dist = np.inf
        _map = {}
        for i in range(cluster.shape[0]):
            f = cluster[i]
            dist = np.linalg.norm(S-f)
            if dist < min_dist:
                min_dist = dist
            _map[dist] = f

        return _map[min_dist]  
        
    def partition_dataset(self, features, labels):
        d = {}
        for j in range(labels.shape[0]):
            d[labels[j]] = []
        for i in range(features.shape[0]):
            feature = features[i]
            label = labels[i]
            d[label].append(list(feature))
        for key in d:
            d[key] = np.array(d[key])

        return d
    
    def build_tree(self, features):
        centroid = self.get_cluster_centroid(features)
        self.set_root_value(centroid)
        
        # Find centroid, set root, and eliminate corresponding row
        for i in range(features.shape[0]):
            if np.all(features[i] == centroid):
                features = np.delete(features, i, axis=0)
                break
        
        if self.L <= 0:
            return
        
        if features.shape[0] < self.bf:
            for c in range(features.shape[0]):
                child_id = INDICES.pop(INDICES.index(min(INDICES))) #(self.K * self.id) + (c + 1)
                self.children[c] = VocabTree(self.bf, self.L - 1, self.k_means_func, features, child_id)
                self.children[c].set_root_value(features[c])
            return
        
        labels = self.KMeans.fit_predict(features)
        clusters = self.partition_dataset(features, labels)
        
        for c in range(len(self.children)):
            if c in labels: # Not always possible to have as many clusters as we want
                child_id = INDICES.pop(INDICES.index(min(INDICES)))
                self.children[c] = VocabTree(self.bf, self.L - 1, self.k_means_func, clusters[c], child_id)
                self.children[c].build_tree(clusters[c])
        
    def _propagate_feature(self, f, code):
        dist_map = {}
        id_map = {}
        for i in range(len(self.children)):
            c = self.children[i]
            if not c:
                return
            dist = np.linalg.norm(f - c.value)
            dist_map[dist] = c.id
            id_map[c.id] = i
        min_id = dist_map[min(dist_map.keys())]
        code[min_id] = code.setdefault(min_id, 0) + 1
        if not self.children[id_map[min_id]].is_leaf_node():
            self.children[id_map[min_id]]._propagate_feature(f, code)
      
    def get_db_size(self, dbdir='./DVDcovers/'):
        ims = [im for im in os.listdir(dbdir) if not im.startswith('.')]
        return len(ims)
    
    def compute_path(self, image):
        code = {}
        _, desc = get_SIFT_descriptors(image.copy())
        for i in range(desc.shape[0]):
            #path = []
            self._propagate_feature(desc[i], code)
            #for j in range(len(path)):
            #    enc = (path[i] + i) * j
            #    code[enc] = code.setdefault(enc, 0) + 1
        return code
    
    def compute_img_paths(self, dbdir='./DVDcovers/'):
        codes = {}
        ims = [im for im in os.listdir(dbdir) if not im.startswith('.')]
        for _imgname in ims:
            imgname = dbdir + _imgname
            img = io.imread(imgname)
            codes[imgname] = self.compute_path(img)
            self.imgdb[imgname] = codes[imgname]
        return codes
    
    def build_image_matrix(self, db_imgcodes):
        imgcodes = {}
        for img in db_imgcodes:
            imgcodes[img] = self._vec_from_dict(db_imgcodes[img])
        imgmatrix = np.vstack(list(imgcodes.values()))
        return imgmatrix
    
    def compute_weight_vector(self, db_imgcodes):
        w = np.zeros((self.num_nodes, ), dtype=np.float64)
        imgmatrix = self.build_image_matrix(db_imgcodes)
        N = imgmatrix.shape[0]
        Ni = 0
        for i in range(imgmatrix.shape[1]):
            codevec = imgmatrix[:, i]
            Ni = np.sum(codevec)
                    
            if Ni == 0:
                w[i] = 0
            else:
                w[i] = np.log(N/Ni)
        
        for imgname in self.imgdb:
            self.imgdb[imgname] = w * self._vec_from_dict(self.imgdb[imgname])
        
        return w
            
    def _vec_from_dict(self, dc):
        d = np.zeros((self.num_nodes, ), dtype=np.float64)
        for k in dc:
            d[k] = dc[k]
        return d
    
    def compute_img_code(self, image):
        imgcode = self.compute_path(image)
        db_imgcodes = self.compute_img_paths()
        w = self.compute_weight_vector(db_imgcodes)
        return w * self._vec_from_dict(imgcode)