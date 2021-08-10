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

import sklearn.cluster
from sklearn.cluster import k_means, MiniBatchKMeans, MeanShift

def get_SIFT_descriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img.copy(), None)
    kp_list = []
    for kp in keypoints:
        kp_list += [(kp.response, kp.pt, kp.size)]
    return kp_list, descriptors
    
def get_ORB_descriptors(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img.copy(), None)
    kp_list = []
    for kp in keypoints:
        kp_list += [(kp.response, kp.pt, kp.size)]
    return kp_list, descriptors

def get_BRISK_descriptors(img):
    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(img.copy(), None)
    kp_list = []
    for i in range(len(keypoints)):
        kp = keypoints[i]
        kp_list += [(kp.response, kp.pt, kp.size, i)]
    return kp_list, descriptors

def get_MSER_descriptors(img):
    img_gray = color.rgb2gray(img)
    blobs = blob_doh(img_gray, max_sigma=30, threshold=.01)
    blob_list = []
    for blob in blobs:
        x, y, r = blob
        blob_list += [(r, (x, y), r)]
    return blob_list, blob_list

_algo_map = {
    'SIFT': get_SIFT_descriptors,
    'ORB': get_ORB_descriptors,
    'BRISK': get_BRISK_descriptors,
}

def visualize_keypoints(img, clr='r', algo='SIFT'):
    # Create a figure. Equal aspect so circles look circular
    fig,ax = plt.subplots(1)
    
    # Only plot n keypoints with highest response for clarity
    kp, desc = _algo_map[algo](img)
    keypoints = sorted(kp)[::-1]

    # Show the image
    ax.imshow(img)
    for kp in keypoints:
        x, y = kp[1]
        plt.scatter(x, y, marker='o', s=kp[2], facecolors='none', edgecolors=clr)
    plt.show()
    
def match_keypoints(img1, img2, algo='SIFT', thresh=0.45):
    kps_1, desc_1 = _algo_map[algo](img1)
    kps_2, desc_2 = _algo_map[algo](img2)
    
    best_matches = []
    for i in range(len(desc_1)):
        src, desc1 = kps_1[i][1], desc_1[i]
        matches = []
        for j in range(len(desc_2)):
            dst, desc2 = kps_2[j][1], desc_2[j]
            sim = np.linalg.norm(desc1-desc2, 2)
            matches += [(float(sim), dst, desc2)]
            
        # get closest match
        indices = [y[0] for y in matches]
        idx = np.argmin(np.array(indices))
        closest = matches[idx]
        
        # get second closest
        matches_without_closest = matches[:]
        matches_without_closest.remove(closest)
        indices = [y[0] for y in matches_without_closest]
        idx2 = np.argmin(np.array(indices))
        sec_closest = matches_without_closest[idx2]
        
        # compute phi
        phi = np.linalg.norm(desc1 - closest[2], 2) / np.linalg.norm(desc1 - sec_closest[2], 2)
        if phi <= thresh:
            best_matches += [(phi, (src, closest[1]))]
            
    
    return best_matches

def get_best_n(matches, n):
    return [x[1] for x in sorted(matches)[::][:n]]

def get_best_n_points(hits, n):
    return [x[-1] for x in sorted(hits)[::][:n]]

def get_best_n_descs(kps, descs, n):
    indices = get_best_n_points(kps, n)
    return descs[indices]

def get_homography_transform(points_from, points_to):
    A = np.ones((8,9))
    
    i = 0
    for j in range(len(points_from)):
        p = np.asarray(points_from[j])
        q = np.asarray(points_to[j])
        A[i] = np.array([p[0], p[1], 1, 0, 0, 0, -q[0]*p[0], -q[0]*p[1], -q[0]])
        A[i+1] = np.array([0, 0, 0, p[0], p[1], 1, -q[1]*p[0], -q[1]*p[1], -q[1]])
        i+=2
    
    hw, hv = np.linalg.eig(A.T @ A)
    idx = np.argmin(hw)
    h = hv[:, idx]
   
    H = np.array([
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], h[8]]
    ])
    
    return H

def draw_matches(img1, img2, matches, size):
    
    import matplotlib.colors as mcolors
    colors = list(mcolors.CSS4_COLORS.values())
    color_map = {}
        
    fig = plt.figure(figsize=size)
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    
    fig.lines = []
    
    for m in matches:
        src, dst = m
        color = random.choice(colors)
        colors.remove(color)
        
        ax1.scatter(src[0], src[1], marker='o', s=size[1]*4, linewidth=2.5, facecolors='none', edgecolors=color)
        ax2.scatter(dst[0], dst[1], marker='o', s=size[1], linewidth=1.5, facecolors='none', edgecolors=color)

    plt.show()
    
def apply_homography(H, p):
    x = (H[0, 0]*p[0] + H[0, 1]*p[1] + H[0, 2]) / (H[2, 0]*p[0] + H[2,1]*p[1] + H[2,2])
    y = (H[1, 0]*p[0] + H[1, 1]*p[1] + H[1, 2]) / (H[2, 0]*p[0] + H[2,1]*p[1] + H[2,2])
    return np.asarray((x,y))

def count_inliers(matches, H, thresh):
    inliers = []
    for m in matches:
        p, q = m
        p = np.asarray(p)
        q = np.asarray(q)
        t = apply_homography(H, p)
        dist = np.linalg.norm(t-q)
        
        if dist < thresh:
            inliers += [t]
     
    plt.show()
    return inliers

def RANSAC_fit_homography(matches, n=5000, thresh=10):
    H = np.zeros((8,9))
    inliers_map = {}
    for i in range(n):  
        # Sample 3 matches at random
        sampled = random.sample(matches, 4)
        points_from = [sample[0] for sample in sampled]
        points_to = [sample[1] for sample in sampled]
        
        # Compute the homography from our samples
        H = get_homography_transform(points_from, points_to)
        
        # Count and save the number of inliers
        inliers = count_inliers(matches, H, thresh=thresh)
        num_inliers = len(inliers)
        inliers_map[num_inliers] = inliers_map.setdefault(num_inliers, []) + [H]
        
    idx = max(list(inliers_map.keys()))
    return inliers_map[idx][-1]

def find_match(img, H, corners):
    warped_corners = []
    for c in corners:
        wc = apply_homography(H, c)
        warped_corners += [wc]

    io.imshow(img)

    for wc in warped_corners:
        x, y = wc
        plt.scatter(x, y, marker='+', color='lime')

def superpose_image(img1, img2):
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if np.sum(img1[i, j]) > 0:
                img2[i, j] = img1[i, j]
    plt.axis("off")
    io.imshow(img2)
    
def build_feature_db(path):
    features = []
    os.chdir('/Users/harma/Desktop/CSC420/p/')
    os.chdir(path)
    db = {}
    for imgname in os.listdir('.'):
        try:
            img = io.imread(imgname)
        except:
            print("Invalid name, skipping: ", imgname)
        kps, desc = get_SIFT_descriptors(img.copy())
        db[imgname] = (kps, desc)
        for row in desc:
            features.append(list(row))
    os.chdir('/Users/harma/Desktop/CSC420/p/')
    return db, np.array(features)

def get_max_id(t):
    if not t:
        return -np.inf
    if t.is_leaf_node():
        return t.id
    else:
        return max(t.id, max(get_max_id(c) for c in t.children))
    
def get_depth(t):
    if not t:
        return 0
    if t.is_leaf_node():
        return 0
    return 1 + max(get_depth(c) for c in t.children)

def L1_score(u, v):
    x = u.copy()
    y = v.copy()
    m, n = x.shape[0], y.shape[0]
    if m < n:
        x = np.hstack([x, np.zeros((n - m,))])
    else: y = np.hstack([y, np.zeros((m - n,))])
    return np.abs(2 + np.sum(np.linalg.norm(x-y) - np.linalg.norm(y) - np.linalg.norm(x)))

def L2_score(u, v):
    x = u.copy()
    y = v.copy()
    m, n = x.shape[0], y.shape[0]
    if m < n:
        x = np.hstack([x, np.zeros((n - m,))])
    else: y = np.hstack([y, np.zeros((m - n,))])        
    return np.linalg.norm(x-y)

def cos_sim(u, v):
    x = u.copy()
    y = v.copy()
    m, n = x.shape[0], y.shape[0]
    if m < n:
        x = np.hstack([x, np.zeros((n - m,))])
    else: y = np.hstack([y, np.zeros((m - n,))])
        
    x *= np.linalg.norm(x)
    y *= np.linalg.norm(y)
        
    from sklearn.metrics.pairwise import cosine_similarity
    return np.sum(cosine_similarity(x.reshape(1,-1), y.reshape(1,-1)))

def compute_results_homographies(images, test_image):
    homographies = []
    inliers_map={}
    for imgname in images:
        img = io.imread(imgname)
        matches = get_best_n(match_keypoints(img, test_image, thresh=0.85), 100)
        H = RANSAC_fit_homography(matches)
        homographies += [H]
        inliers_map[imgname] = count_inliers(matches, H, thresh=10)
        
    warps = []
    imgs = [io.imread(imgname) for imgname in images]
    for i in range(len(homographies)):
        warped = cv2.warpPerspective(imgs[i], homographies[i], (1000,750))
        warps += [warped]
        
    return warps, homographies, inliers_map