import pdb
import random
import numpy as np
import os, json

from random import shuffle
def shuffle_list(*ls,seed=0):
    random.seed(seed)
    l = list(zip(*ls))
    shuffle(l)
    return zip(*l)

from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn

import h5py
import pandas as pd


def filter_df(df, filter_dict):
    if len(filter_dict) > 0:
        filter_mask = np.full(len(df), True, bool)
        # assert 'label' not in filter_dict.keys()
        for key, val in filter_dict.items():
            mask = df[key].isin(val)
            filter_mask = np.logical_and(filter_mask, mask)
        df = df[filter_mask]
    return df

def df_prep(data, label_dict, ignore, label_col):
    if label_col != 'label':
        data['label'] = data[label_col].copy()

    mask = data['label'].isin(ignore)
    data = data[~mask]
    data.reset_index(drop=True, inplace=True)
    for i in data.index:
        key = data.loc[i, 'label']
        data.at[i, 'label'] = label_dict[key]

    return data

def get_split_from_df(slide_data, all_splits, split_key='train'):
    split = all_splits[split_key]
    split = split.dropna().reset_index(drop=True)

    if len(split) > 0:
        # pdb.set_trace()
        mask = slide_data['slide_id'].isin(split.tolist())
        df_slice = slide_data[mask].reset_index(drop=True)
        # split = Generic_IMG_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes, train_eval=split_key)
        return df_slice

    else:
        return None

def get_data_list(data_dir,train_val_test, splitter_path, label_csv_path):
    # load
    slide_data = pd.read_csv(label_csv_path)
    # pdb.set_trace()
    slide_data = filter_df(slide_data, {})
    slide_data = df_prep(slide_data, label_dict={'IDC': 0, 'ILC': 1}, ignore=[], label_col='label')
    all_splits = pd.read_csv(splitter_path, dtype=slide_data['slide_id'].dtype)
    #
    df_slice = get_split_from_df(slide_data, all_splits, split_key=train_val_test)
    slide_id_list = df_slice['slide_id'].tolist()
    label_list_ori = df_slice['label'].tolist()

    wsi_list, label_list = [], []
    pdb.set_trace()

    wsi_bags_topK_info = os.path.join(data_dir, 'resnet34_clam_attention_top1024_train_info.json')
    if os.path.exists(wsi_bags_topK_info):
        with open(wsi_bags_topK_info, 'r') as myfile:
            if len(myfile.readlines()) != 0:
                myfile.seek(0)
                info = json.load(myfile)
                wsi_list, label_list = info['wsi_list'], info['label_list']
                return wsi_list, label_list
    topK_dict_info = os.path.join(data_dir, 'resnet34_clam_attention_top1024_info.json')
    with open(topK_dict_info, 'r') as myfile:
        if len(myfile.readlines()) != 0:
            myfile.seek(0)
            topK_dict = json.load(myfile)

    for i, slide in enumerate(slide_id_list):
        wsi_path = os.path.join(data_dir,slide)
        if os.path.exists(wsi_path):
            instance_bag_list = os.listdir(wsi_path)
            topK_instances_list = []
            for i, instance_name in enumerate(instance_bag_list):
                if instance_name not in topK_dict[slide]:continue
                instance_path = os.path.join(wsi_path, instance_name)
                topK_instances_list.append(instance_path)

            wsi_list.append(topK_instances_list)
            label_list.append(label_list_ori[i])

    info_dict = {'wsi_list':wsi_list,'label_list':label_list}
    with open(wsi_bags_topK_info, 'w', encoding='utf-8') as f:
        json.dump(info_dict, f)

    return wsi_list, label_list

class ExternalInputIterator(object):
    def __init__(self, data_dir, batch_size, splitter_path, shuffle=False, device_id=0, num_gpus=1, train_eval_test='val',
                 bag_size=1024,label_csv_path='/data1/lhl_workspace/CLAM-master/dataset_csv/camelyon16.csv'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_val_test = train_eval_test
        self.bag_size = bag_size
        self.num_gpus = num_gpus
        self.device_id = device_id
        self.wsi_list_all, self.label_list_all = get_data_list(data_dir,self.train_val_test, splitter_path,label_csv_path=label_csv_path)
        if self.shuffle:
            self.wsi_list_all, self.label_list_all = shuffle_list(self.wsi_list_all, self.label_list_all)

        temp_dataset_len = len(self.wsi_list_all)

        self.wsi_list = self.wsi_list_all[temp_dataset_len // num_gpus * device_id:
                                 temp_dataset_len // num_gpus * (device_id + 1)]

        self.label_list = self.label_list_all[temp_dataset_len // num_gpus * device_id:
                                     temp_dataset_len // num_gpus * (device_id + 1)]
        self.data_set_len = len(self.wsi_list)
        self.index_list = range(self.data_set_len)

        self.n = len(self.wsi_list)


    def __len__(self):
        return len(self.wsi_list)


    def __iter__(self):
        self.i = 0
        print('re iter')
        if self.shuffle:
            print('shuffle all data?')
            self.wsi_list_all, self.label_list_all = shuffle_list(self.wsi_list_all, self.label_list_all)

            temp_dataset_len = len(self.wsi_list_all)

            self.wsi_list = self.wsi_list_all[temp_dataset_len // self.num_gpus * self.device_id:
                                              temp_dataset_len // self.num_gpus * (self.device_id + 1)]
            self.label_list = self.label_list_all[temp_dataset_len // self.num_gpus * self.device_id:
                                                  temp_dataset_len // self.num_gpus * (self.device_id + 1)]
            self.data_set_len = len(self.wsi_list)
            self.index_list = range(self.data_set_len)

            self.n = len(self.wsi_list)
        return self

    def __next__(self):
        batch = []
        labels = []
        if self.i >= self.n:
            if True:
                print('shuffle all data?')
                self.wsi_list_all, self.label_list_all = shuffle_list(self.wsi_list_all, self.label_list_all)

                temp_dataset_len = len(self.wsi_list_all)

                self.wsi_list = self.wsi_list_all[temp_dataset_len // self.num_gpus * self.device_id:
                                                  temp_dataset_len // self.num_gpus * (self.device_id + 1)]
                self.label_list = self.label_list_all[temp_dataset_len // self.num_gpus * self.device_id:
                                                      temp_dataset_len // self.num_gpus * (self.device_id + 1)]
                self.data_set_len = len(self.wsi_list)
                self.index_list = range(self.data_set_len)

                self.n = len(self.wsi_list)
            raise StopIteration

        for _ in range(self.batch_size):
            sample_idx = self.i
            full_path = self.wsi_list[sample_idx]

            with h5py.File(full_path, 'r') as hdf5_file:
                patches = hdf5_file['patches'][:]
                # coords = hdf5_file['coords'][:]
            j = 0
            for j,patch in enumerate(patches):
                if j >= self.bag_size: break
                batch.append(patch)
                labels.append(np.array([self.label_list[sample_idx],0], dtype=np.uint8))
            while j < self.bag_size - 1:
                batch.append(batch[-1])
                labels.append(np.array([self.label_list[sample_idx],1], dtype=np.uint8))
                j += 1
                # print('not long enough',j)
            self.i = (self.i + 1) % self.n
        return (batch, labels, )

    @property
    def size(self, ):
        return self.data_set_len

    next = __next__
class ExternalInputCallable:
    def __init__(self, data_dir, batch_size, splitter_path, shuffle=False, device_id=0, num_gpus=1, train_eval_test='val',
                 bag_size=1024,label_csv_path='/data1/lhl_workspace/CLAM-master/dataset_csv/camelyon16.csv'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_val_test = train_eval_test
        self.bag_size = bag_size
        self.num_gpus = num_gpus
        self.device_id = device_id
        self.wsi_list_all, self.label_list_all = get_data_list(data_dir,self.train_val_test, splitter_path,label_csv_path=label_csv_path)
        if self.shuffle:
            self.wsi_list_all, self.label_list_all = shuffle_list(self.wsi_list_all, self.label_list_all)

        temp_dataset_len = len(self.wsi_list_all)

        self.wsi_list = self.wsi_list_all[temp_dataset_len // num_gpus * device_id:
                                 temp_dataset_len // num_gpus * (device_id + 1)]

        self.label_list = self.label_list_all[temp_dataset_len // num_gpus * device_id:
                                     temp_dataset_len // num_gpus * (device_id + 1)]
        self.data_set_len = len(self.wsi_list)
        self.index_list = range(self.data_set_len)

        self.n = len(self.wsi_list)
        self.full_iterations = len(self.wsi_list) // batch_size

        self.perm = None  # permutation of indices
        self.last_seen_epoch = None  # so that we don't have to recompute the `self.perm` for every sample
        self.i = 0

    def __call__(self, sample_info):
        if sample_info >= self.full_iterations:
            # self.perm = np.random.default_rng(seed=2021).permutation(len(self.wsi_list))
            if self.shuffle:
                self.wsi_list_all, self.label_list_all = shuffle_list(self.wsi_list_all, self.label_list_all)
                self.wsi_list, self.label_list = self.wsi_list_all, self.label_list_all
            raise StopIteration

        batch, labels = [], []
        for _ in range(self.batch_size):
            sample_idx = sample_info
            wsi_patches_list = self.wsi_list[sample_idx]
            j = 0
            for j, filename in enumerate(wsi_patches_list):
                if j >= self.bag_size: break
                try:
                    f = open(filename, 'rb')
                except:
                    print('%%Loading Error:',filename)
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
                # batch.append(self.wsi_list)
                labels.append(np.array([self.label_list[sample_idx], 0], dtype=np.uint8))
            while j < self.bag_size - 1:
                batch.append(batch[-1])
                labels.append(np.array([self.label_list[sample_idx], 1], dtype=np.uint8))
                j += 1
                # print('not long enough',j)
            # self.i = (self.i + 1) % self.n
        return (batch, labels,)

    @property
    def size(self, ):
        return self.data_set_len
def TrainPipeline(eii, batch_size, num_threads, device_id, seed, img_size=256, use_h5=False):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    device = 'gpu'
    with pipe:
        if use_h5:
            images, labels = fn.external_source(source=eii, num_outputs=2)
        else:
            jpegs, labels = fn.external_source(source=eii, num_outputs=2)
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        if device == 'gpu':
            images=images.gpu()
        # hsv
        # random_prob = fn.random.coin_flip(probability=0.2)
        # fn.cast(random_prob, dtype=types.FLOAT)
        # if True:
        #     images = fn.hsv(images.gpu(), hue=random_prob * fn.random.uniform(range=(0.0, 359.0)),
        #                                 saturation=random_prob * fn.random.uniform(range=(0.8, 1.2)),
        #                                 value=random_prob * fn.random.uniform(range=(0.8, 1.2)),device='gpu')
        # brightness
        images = fn.brightness_contrast(images, contrast=fn.random.uniform(range=(0.8, 1.2)),
                                        brightness=fn.random.uniform(range=(0.8, 1.2)),device=device)
        # rotate
        images = fn.rotate(images, keep_size=True, angle = fn.random.uniform(range=(-180.0, 180.0)), fill_value=0, device=device)

        # random resized_crop
        images = fn.random_resized_crop(images, size=(img_size, img_size), random_area=[0.7,1.0], device=device)
        # images = fn.resize(images, size=img_size)
        # random flip
        images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5),
                         vertical=fn.random.coin_flip(probability=0.5), device=device)

        # scale to 0~1.0
        images = images / 255.0
        # normalize
        images = fn.normalize(images, device=device, axes=[0,1],mean=np.array([0.485, 0.456, 0.406]).reshape((1,1,3)),
                              stddev=np.array([0.229, 0.224, 0.225]).reshape((1,1,3)))

        images = fn.transpose(images, device=device, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe

def ValPipeline(eii, batch_size, num_threads, device_id, seed, img_size=256, use_h5=False, use_gpu=True):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=seed)
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    with pipe:
        if use_h5:
            images, labels = fn.external_source(source=eii, num_outputs=2)
        else:
            jpegs, labels = fn.external_source(source=eii, num_outputs=2)
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        # scale to 0~1.0
        if device == 'gpu':
            images=images.gpu()
        images = images / 255.0
        # normalize
        # images = fn.resize(images,size=img_size)
        images = fn.normalize(images, device=device, axes=[0, 1],
                              mean=np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
                              stddev=np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
        images = fn.transpose(images, device=device, perm=[2, 0, 1])
        pipe.set_outputs(images, labels)
    return pipe

def get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',
                   splitter_path='./', device_id=0, num_gpus=1, seed=1, bag_size=1024,label_csv_path='./'):
    eii = ExternalInputCallable(data_dir=data_dir, batch_size=batch_size,
                                splitter_path=splitter_path, shuffle=shuffle,
                                device_id=device_id, num_gpus=num_gpus, train_eval_test=train_eval_test, bag_size=bag_size,
                                label_csv_path=label_csv_path)
    if train_eval_test=='train':
        pipe = TrainPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                                seed=seed + device_id)
    elif train_eval_test=='val':
        pipe = ValPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                              seed=seed + device_id)
    else:
        pipe = ValPipeline(batch_size=batch_size * bag_size, eii=eii, num_threads=num_threads, device_id=device_id,
                           seed=seed + device_id,use_gpu=False)
    pipe.build()
    loader = DALIClassificationIterator(pipe, size=eii.size * bag_size,auto_reset=True,
                                        last_batch_padded=True,prepare_first_batch=False)

    return loader

def get_train_val_test_loaders(data_dir,splitter_path, bag_size=256, label_csv_path='./'):
    train_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=True, num_threads=2, train_eval_test='train',
                                  splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    val_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    test_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='test',
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=bag_size,label_csv_path=label_csv_path)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = '/data1/lhl_workspace/CLAM-master/topk_rois/v2/'
    train_val_test= 'train'
    splitter_path = "/data5/kww/CLAM/splits/task_1_tumor_vs_normal_100/splits_0.csv"
    label_csv_path = '/data1/lhl_workspace/CLAM-master/dataset_csv/camelyon16.csv'

    # x = get_data_list(data_dir,train_val_test, splitter_path,label_csv_path)
    # print(x,len(x[0]))
    train_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=True, num_threads=2, train_eval_test='train',
                   splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    print(len(train_loader))

    val_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='val',
                                  splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    test_loader = get_wsi_loader(data_dir, batch_size=1, shuffle=False, num_threads=2, train_eval_test='test',
                                splitter_path=splitter_path, device_id=0, num_gpus=1, seed=1, bag_size=1024)
    # print(train_loader)
    # test_loader.reset()
    pdb.set_trace()

