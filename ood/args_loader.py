import argparse
import os

def get_args(root):
    parser = argparse.ArgumentParser(description='OOD detection')

    # learning setting
    parser.add_argument('--train_bs', default=16, type=int, help='training batch size')
    parser.add_argument('--gpu', default=0, type=int, help='gpu number')
    parser.add_argument('--timestamp', default=None, help='timestamp')

    # data path
    parser.add_argument('--train_data', default='./data/SeaShips', type=str)
    parser.add_argument('--patch_size', default=128, type=int, help='patch size')

    # model
    parser.add_argument('--backbone_arch', default='resnet50', choices=['resnet50', 'resnet101', 'efficientnet_v2', 'mobilenet_v3'], type=str, help='')

    # cluster
    parser.add_argument('--cluster', default='BisectingKMeans', type=str, choices=['KMeans', 'GaussianMixture', 'MeanShift', 'AffinityPropagation', 'HDBSCAN', 'HDBSCANm', 'BisectingKMeans'])
    parser.add_argument('--num_cluster', default=30, type=int, help='how many k-menas clusters')

    # score
    parser.add_argument('--score_metric', default='cosineSim', type=str, choices=['euclidean', 'mahalanobis', 'cosineSim'])

    args = parser.parse_args()

    # save path
    train_data_name = args.train_data.split('/')[-1]
    if args.timestamp not in [None, 'None', 'none']:
        train_data_name = args.timestamp

    args.train_feat_path = os.path.join(root, 'ood/cache/feature/', f'train_{args.backbone_arch}_s{args.patch_size}_{train_data_name}.pkl')
    args.cluster_path = os.path.join(root, 'ood/cache/cluster/', f'{args.cluster}_k{args.num_cluster}_{args.backbone_arch}_s{args.patch_size}_{train_data_name}.pkl')
    args.distribution_path = os.path.join(root, 'ood/cache/distribution/', f'{args.cluster}_k{args.num_cluster}_{args.backbone_arch}_s{args.patch_size}_{train_data_name}.pkl')
    args.threshold_path = os.path.join(root, 'ood/cache/threshold/', f'{args.cluster}_k{args.num_cluster}_{args.backbone_arch}_s{args.patch_size}_{train_data_name}.json')
    args.score_path = os.path.join(root, 'ood/scores/', f'{args.cluster}_k{args.num_cluster}_{args.backbone_arch}_{train_data_name}_s{args.patch_size}.pkl')

    return args