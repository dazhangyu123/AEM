import pandas as pd
import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
from utils.utils import collate_features
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from models import build_model, build_model_vlm

import h5py
import openslide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Extracting instance features')
parser.add_argument('--dataset', type=str, default='bracs')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--pretrain', default='plip', choices=['natural_supervised', 'medical_ssl', 'plip', 'virchow',
                    'path-clip-B', 'path-clip-L-336', 'openai-clip-B', 'openai-clip-L-336', 'quilt-net', 'biomedclip', 'path-clip-L-768', 'UNI', 'GigaPath'],
                    help='pretrained encoder')
args = parser.parse_args()


@torch.no_grad()
def extract_feature(file_path, wsi, model, pretrain,
                     batch_size=8, verbose=0):
    """
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrain=pretrain)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=16, collate_fn=collate_features)

    if verbose > 0:
        print('processing {}: total of {} batches'.format(file_path, len(loader)))

    feature_list = []
    coord_list = []
    for count, (batch, coords) in enumerate(loader):
        batch = batch.to(device, dtype=torch.float32)
        if pretrain == 'plip' or pretrain == 'quilt-net':
            feature = model.get_image_features(batch)
        elif pretrain == 'UNI' or pretrain == 'GigaPath' or pretrain == 'virchow' or\
                pretrain == 'medical_ssl' or pretrain == 'natural_supervised':
            feature = model(batch)
        else:
            feature = model.encode_image(batch)
        feature_list.append(feature.cpu())
        coord_list.append(coords)
    features = torch.cat(feature_list, dim=0)
    coords = np.concatenate(coord_list, axis=0)

    return features.numpy(), coords



if __name__ == '__main__':
    root_dir = '/mnt/Xsky/zyl/dataset'
    if args.dataset == 'camelyon':
        args.data_h5_dir = os.path.join(root_dir, 'CAMELYON16/coords_anno')
        args.data_slide_dir = os.path.join(root_dir, 'CAMELYON16/training')
        args.csv_path = 'dataset_csv/camelyon16.csv'
        args.slide_ext = '.tif'
        args.data_dir = '/mnt/Xsky/zyl/dataset/CAMELYON16/roi_feats'
    elif args.dataset == 'camelyon17':
        args.data_h5_dir = os.path.join(root_dir, 'CAMELYON17/coords_anno')
        args.data_slide_dir = os.path.join(root_dir, 'CAMELYON17/images')
        args.csv_path = 'dataset_csv/camelyon17.csv'
        args.slide_ext = '.tif'
        args.data_dir = '/mnt/Xsky/zyl/dataset/CAMELYON17/roi_feats'
    elif args.dataset == 'bracs':
        args.data_h5_dir = os.path.join(root_dir, 'bracs/coords_anno_x20')
        args.data_slide_dir = '/mnt/Xsky/bracs/BRACS_WSI'
        args.csv_path = 'dataset_csv/bracs.csv'
        args.slide_ext = '.svs'
        args.data_dir = '/mnt/Xsky/zyl/dataset/bracs/roi_feats_x20'
    elif args.dataset == 'tcga':
        args.data_h5_dir = os.path.join(root_dir, 'bracs/coords_anno_x20')
        args.data_slide_dir = '/mnt/Xsky/bracs/BRACS_WSI'
        args.csv_path = 'dataset_csv/bracs.csv'
        args.slide_ext = '.svs'
        args.data_dir = '/mnt/Xsky/zyl/dataset/bracs/roi_feats_x20'
    else:
        print(f"Dataset %s is not found"%args.dataset)
        exit()

    os.makedirs(args.data_dir, exist_ok=True)

    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError
    df = pd.read_csv(csv_path)


    print('loading model checkpoint')
    if args.pretrain != 'medical_ssl' and args.pretrain != 'natural_supervised' and args.pretrain != 'UNI' and \
            args.pretrain != 'GigaPath' and args.pretrain != 'virchow':
        model = build_model_vlm(args)
    else:
        model = build_model(args)
    model = model.to(device)
    model.eval()
    total = len(df)

    output_path = os.path.join(args.data_dir, 'patch_feats_pretrain_%s.h5'%args.pretrain)
    h5file = h5py.File(output_path, "w")
    for bag_candidate_idx in range(total):
        slide_id = df.loc[bag_candidate_idx, 'slide_id']
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            continue
        if args.dataset != 'bracs':
            slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        else:
            slide_file_path = df.loc[bag_candidate_idx, 'full_path']
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)
        slide_feature, coords = extract_feature(h5_file_path, wsi,
                                            model=model, pretrain=args.pretrain, batch_size=args.batch_size, verbose=1)

        slide_grp = h5file.create_group(slide_id)
        slide_grp.create_dataset('feat', data=slide_feature.astype(np.float16))
        slide_grp.create_dataset('coords', data=coords)
        slide_grp.attrs['label'] = df.loc[bag_candidate_idx, 'label']
        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(slide_id, time_elapsed))

    h5file.close()
    print("Stored features successfully!")