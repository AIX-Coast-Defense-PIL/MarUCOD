import torch
import torchvision
import torch.backends.cudnn as cudnn

import os
import numpy as np
import random
import time

from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, HDBSCAN, BisectingKMeans
from sklearn.mixture import GaussianMixture

import sys 
work_path = os.path.join(os.getcwd(), os.pardir) if '/ood' in os.getcwd() else os.getcwd()
work_path = work_path if 'Yolo-Seg-OOD' in work_path else os.path.join(work_path, 'Yolo-Seg-OOD')
sys.path.append(work_path)

from utils.file_processing import save_file

from args_loader import get_args
from data_loader import get_train_loader
from ood_scores import calc_distance_score
from calculate import calc_distributions

import warnings
warnings.filterwarnings('ignore')


def seed_setting(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)
    np.random.seed(num)
    random.seed(num)
    cudnn.deterministic = True

def load_model(args):
    if args.backbone_arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.backbone_arch == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif args.backbone_arch == 'efficientnet_v2':
        model = torchvision.models.efficientnet_v2_m(pretrained=True)
    elif args.backbone_arch == 'mobilenet_v3':
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
    else:
        print('wrong backbone_arch')
    
    state_dict = {k:model.state_dict()[k] for k in model.state_dict() if not k in ['fc.bias', 'fc.weight']}
    model.load_state_dict(state_dict, strict=False)

    return model

def get_bbox_infos(args, feat_path, model):
    data_dir_paths = [args.train_data]
    data_loader = get_train_loader(data_dir_paths, batch_size=args.train_bs, patch_size=args.patch_size)


    model.to(device)
    model.eval()

    feat_infos = []
    pred_infos = []
    start_time = time.time()
    with torch.no_grad():
        print('start bbox info extracting...')
        for batch_idx, (images, img_names, labels, bboxes) in enumerate(data_loader):
            start_batch_time = time.time()

            images = images.to(device)
            feats = model(images)
            feats = feats.data.cpu().numpy()
            
            for idx in range(len(labels)):
                feat_infos.append({'img_name':img_names[idx], 'label':labels[idx], 'bbox':[int(bboxes[coordicate_idx][idx]) for coordicate_idx in range(4)], 'feat':feats[idx]})

            if batch_idx % 20 == 0:
                batch_time = time.time() - start_batch_time
                print(f'[{batch_idx}/{len(data_loader)} batch] {batch_time:.04f} sec per batch')
            
            # memory management
            del feats
            del labels
            torch.cuda.empty_cache()
    total_time = time.time() - start_time
    
    save_file(feat_infos, feat_path)
    time_per_img = total_time / len(feat_infos)
    print(f'[Saved the features] path : {feat_path}, total time : {total_time:.4f}, time : {time_per_img:.4f} s/f')
    
    return feat_infos, pred_infos
        
def train_cluster(args, feats):

    start_time = time.time()
    if args.cluster == 'KMeans':
        cluster = KMeans(n_clusters=args.num_cluster).fit(feats) if args.num_cluster!=0 else KMeans().fit(feats)
    elif args.cluster == 'GaussianMixture':
        cluster = GaussianMixture(n_components=args.num_cluster, max_iter=200).fit(feats) if args.num_cluster!=0 else GaussianMixture().fit(feats)
    elif args.cluster == 'MeanShift':
        cluster = MeanShift(bandwidth=args.num_cluster).fit(feats) if args.num_cluster!=0 else MeanShift().fit(feats)
    elif args.cluster == 'AffinityPropagation':
        cluster = AffinityPropagation(random_state=args.num_cluster).fit(feats) if args.num_cluster!=0 else AffinityPropagation().fit(feats)
    elif args.cluster == 'HDBSCAN':
        cluster = HDBSCAN(min_cluster_size=args.num_cluster, store_centers='centroid').fit(feats) if args.num_cluster!=0 else HDBSCAN(store_centers='both').fit(feats)
    elif args.cluster == 'HDBSCANm':
        cluster = HDBSCAN(min_cluster_size=args.num_cluster, store_centers='medoid').fit(feats) if args.num_cluster!=0 else HDBSCAN(store_centers='medoid').fit(feats)
    elif args.cluster == 'BisectingKMeans':
        cluster = BisectingKMeans(n_clusters=args.num_cluster, random_state=0).fit(feats) if args.num_cluster!=0 else BisectingKMeans().fit(feats)
    
    total_time = time.time() - start_time
    print(f'time to fitting cluster : {total_time:.4f} s')

    # save cluster model
    save_file(cluster, args.cluster_path)

    return cluster

def ood_train(args):
    print('start ood training...')
    backbone = load_model(args)
    
    feat_infos, _ = get_bbox_infos(args, args.train_feat_path, backbone)
    feats = [feat_info['feat'] for feat_info in feat_infos]

    print('calcuating clusters...')
    cluster = train_cluster(args, feats)
    print(f'[saved {args.num_cluster} cluster model] path :', args.cluster_path)

    print('calcuating train distribution...')
    distributions = calc_distributions(args.cluster, args.score_metric, cluster, feats)
    save_file(distributions, args.distribution_path)
    print('[saved train distribution] path :', args.distribution_path)

    # find thresholds
    print('calculating ood thresholds...')
    ood_scores = calc_distance_score(distributions, feats, args.score_metric)
    ood_scores = sorted(ood_scores, reverse=True)
    thresholds_in_rate = {str(in_rate)+"%" : round(np.float64(ood_scores[round((1-(in_rate*0.01)) * len(ood_scores))]), 2) for in_rate in range(1, 101, 1)}

    save_file(thresholds_in_rate, args.threshold_path)
    print('[saved ood thresholds] path :', args.threshold_path)

    return thresholds_in_rate

if __name__=='__main__':
    args = get_args(work_path)

    print(f"Start OOD cluster ({args.cluster})! \n")    
    device = torch.device(f'cuda:{args.gpu}')
    seed_setting(0)

    thresholds_in_rate = ood_train(args)
    print(f'ood thresholds : {thresholds_in_rate}')
    
    print(f"\nOOD cluster ({args.cluster}) Done! \n")
