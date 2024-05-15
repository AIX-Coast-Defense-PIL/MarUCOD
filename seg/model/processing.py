import cv2
import numpy as np

def ground_sky_filtering(seg_pred, img0_shape=None, surr_sky_ratio_eps = 0.3, width_eps = 0.7, dilate_eps = 5):
    # seg_pred(h, w) consists of {0,1,2} (0 = obstacles, 1 = water, 2 = sky)
    # binary mask (obstacle=1, others(sky,sea)=0)
    seg_pred = np.where(seg_pred==0,4,seg_pred) # (4 = obstacles, 1 = water, 2 = sky)
    obs_bin = np.uint8(np.where(seg_pred==4,1,0))

    ## Connected Component Labeling 
    ccl_num_labels, ccl_labels, ccl_stats, ccl_centroids = cv2.connectedComponentsWithStats(obs_bin)

    ## Ground Extraction
    ground_sky_bin = np.uint8(np.where(seg_pred==2,1,0)) # binary mask for sky components
    for ccl_idx, (x,y,w,h,a) in enumerate(ccl_stats):
        if ccl_idx==0: continue

        ccl_bin = np.uint8(np.where(ccl_labels==ccl_idx, 1, 0))
        surr_pix = cv2.dilate(ccl_bin, np.ones((dilate_eps,dilate_eps), np.uint8), iterations=1) - ccl_bin
        sum_pix = surr_pix.sum()
        surr_pix = np.multiply(seg_pred, surr_pix.astype(int))
        if surr_pix.sum() != 0:
            if ((surr_pix==2).sum() / sum_pix > surr_sky_ratio_eps) \
                or (w > obs_bin.shape[1]*width_eps) or (h > obs_bin.shape[0]*width_eps): 
                ground_sky_bin += ccl_bin
        
    ground_sky_bin = np.uint8(np.where(ground_sky_bin>0, 1, 0))

    if img0_shape is not None:
        ground_sky_bin = cv2.resize(ground_sky_bin.astype('float32'), img0_shape[::-1])
        ground_sky_bin = np.uint8(np.where(ground_sky_bin>=0.5, 1, 0)) # mask binary
    
    return np.uint8(ground_sky_bin)