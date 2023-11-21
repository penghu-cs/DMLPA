import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import numpy as np
import scipy.io as sio
import h5py 

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = [d[index] for d in self.data]
        labels = [d[index] for d in self.labels]
        return data, labels

    def __len__(self):
        count = len(self.data[0])
        return count


class SingleModalDataSet(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        count = len(self.data)
        return count

data_names = ['pascal', 'xmedianet', 'MS-COCO']
def get_ImgTxt(dataset):
    if 'pascal' in dataset:
        dataset = './data/pascal/pascal_deep_idx_data-corr-ae.h5py'
    elif 'xmedianet' in dataset:
        dataset = './data/XMediaNet/xmedianet_deep_idx_data_has_valid.h5py'
    elif 'MS-COCO' in dataset:
        dataset = './data/MSCOCO/MSCOCO_deep_idx_data_3subset_1.h5py'
    import pdb
    h = h5py.File(dataset)
    wv_matrix = h['wv_matrix'][()]
    img_train = h['train_imgs_deep'][()].astype('float32')
    # text_train = wv_matrix[h['train_texts_idx'][()].astype('int64')].sum(1, keepdims=True).reshape([img_train.shape[0], -1]) # h['train_texts_idx'][()].astype('int64') 
    text_train = h['train_texts_idx'][()].astype('int64') 
    label_imgs_train = h['train_imgs_labels'][()]
    label_txts_train = h['train_texts_labels'][()]

    img_valid = h['valid_imgs_deep'][()].astype('float32')
    # text_valid = wv_matrix[h['valid_texts_idx'][()].astype('int64')].sum(1, keepdims=True).reshape([img_valid.shape[0], -1]) # h['valid_texts_idx'][()].astype('int64')
    text_valid = h['valid_texts_idx'][()].astype('int64')
    label_imgs_valid = h['valid_imgs_labels'][()]
    label_txts_valid = h['valid_texts_labels'][()]

    img_test = h['test_imgs_deep'][()].astype('float32')
    # text_test = wv_matrix[h['test_texts_idx'][()].astype('int64')].sum(1, keepdims=True).reshape([img_test.shape[0], -1]) # h['test_texts_idx'][()].astype('int64')
    text_test = h['test_texts_idx'][()].astype('int64')
    label_imgs_test = h['test_imgs_labels'][()]
    label_txts_test = h['test_texts_labels'][()]

    h.close()
    if len(label_imgs_train.shape) == 1 or label_imgs_train.shape[1] == 1:
        label_imgs_train = label_imgs_train - label_imgs_train.min()
        label_txts_train = label_imgs_train - label_imgs_train.min()
        label_imgs_test = label_imgs_test - label_imgs_train.min()
        label_txts_test = label_txts_test - label_imgs_train.min()
        label_imgs_valid = label_imgs_valid - label_imgs_train.min()
        label_txts_valid = label_txts_valid - label_imgs_train.min()

        max_lab = label_imgs_train.max()
        label_imgs_train = (label_imgs_train.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_train = (label_txts_train.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_imgs_test = (label_imgs_test.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_test = (label_txts_test.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_imgs_valid = (label_imgs_valid.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_valid = (label_txts_valid.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        
    img_train, text_train, img_valid, text_valid, img_test, text_test = torch.tensor(img_train).float(), torch.tensor(text_train), torch.tensor(img_valid).float(), torch.tensor(text_valid), torch.tensor(img_test).float(), torch.tensor(text_test)
    label_imgs_train, label_txts_train = torch.tensor(label_imgs_train.astype(int)), torch.tensor(label_txts_train.astype(int))
    label_imgs_test, label_txts_test = torch.tensor(label_imgs_test.astype(int)), torch.tensor(label_txts_test.astype(int))
    label_imgs_valid, label_txts_valid = torch.tensor(label_imgs_valid.astype(int)), torch.tensor(label_txts_valid.astype(int))
    
    inx = label_imgs_train.sum(0).nonzero().reshape([-1])
    label_imgs_train, label_txts_train = label_imgs_train[:, inx], label_txts_train[:, inx]
    label_imgs_test, label_txts_test = label_imgs_test[:, inx], label_txts_test[:, inx]
    label_imgs_valid, label_txts_valid = label_imgs_valid[:, inx], label_txts_valid[:, inx]
    

    data = {'train': [img_train, text_train], 'valid': [img_valid, text_valid], 'test': [img_test, text_test]}
    labels = {'train': [label_imgs_train, label_txts_train], 'valid': [label_imgs_valid, label_txts_valid], 'test': [label_imgs_test, label_txts_test]}
    
    # wv_matrix = None
    return data, labels, wv_matrix

def get_Reuters():
    reuters = sio.loadmat('./dataReuters/Reuters_PCA_1024.mat')
    train_data = [reuters['train'][0, i].astype('float32').T for i in range(reuters['train'].shape[1])]
    min_shift = reuters['train_labels'][0, 0].astype('int64').min()
    train_labels = [reuters['train_labels'][0, i].astype('int64').reshape([-1]) - min_shift for i in range(reuters['train_labels'].shape[1])]
    valid_data = [reuters['valid'][0, i].astype('float32').T for i in range(reuters['valid'].shape[1])]
    valid_labels = [reuters['valid_labels'][0, i].astype('int64').reshape([-1]) - min_shift for i in range(reuters['valid_labels'].shape[1])]
    test_data = [reuters['test'][0, i].astype('float32').T for i in range(reuters['test'].shape[1])]
    test_labels = [reuters['test_labels'][0, i].astype('int64').reshape([-1]) - min_shift for i in range(reuters['test_labels'].shape[1])]
    
    if len(label_imgs_train.shape) == 1 or label_imgs_train.shape[1] == 1:
        label_imgs_train = label_imgs_train - label_imgs_train.min()
        label_txts_train = label_imgs_train - label_imgs_train.min()
        label_imgs_test = label_imgs_test - label_imgs_train.min()
        label_txts_test = label_txts_test - label_imgs_train.min()
        label_imgs_valid = label_imgs_valid - label_imgs_train.min()
        label_txts_valid = label_txts_valid - label_imgs_train.min()

        max_lab = label_imgs_train.max()
        label_imgs_train = (label_imgs_train.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_train = (label_txts_train.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_imgs_test = (label_imgs_test.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_test = (label_txts_test.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_imgs_valid = (label_imgs_valid.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
        label_txts_valid = (label_txts_valid.reshape([-1, 1]) == np.arange(max_lab + 1).reshape([1, -1])).astype(int)
    data = {'train': train_data, 'valid': valid_data, 'test': test_data}
    labels = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}
    return data, labels

def get_nMSAD ():
    mnist_spoken = sio.loadmat('./dataSAD/mnist_spoken_data.mat')
    shuffle_inx = mnist_spoken['shuffle_inx'][0] - 1
    data = mnist_spoken['data'][0]
    labels = mnist_spoken['labels'][0]
    
    for i in range(len(labels)):
        a = (labels[i] - labels[i].min()).reshape([-1,])
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1
        labels[i] = b

    train_num = 4000
    valid_num = 800
    train_inx = shuffle_inx[0: train_num]
    valid_inx = shuffle_inx[train_num: train_num + valid_num]
    test_inx = shuffle_inx[train_num + valid_num::]
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = [], [], [], [], [], []
    for i in range(len(data)):
        if (data[i].shape[1] % 28) == 0:
            dim = 28
        else:
            dim = 25
        train_data.append(data[i][train_inx].reshape([train_inx.shape[0], 1, dim, -1], order='F').astype('float32'))
        train_labels.append((labels[i][train_inx]).astype('float32'))

        valid_data.append(data[i][valid_inx].reshape([valid_inx.shape[0], 1, dim, -1], order='F').astype('float32'))
        valid_labels.append((labels[i][valid_inx]).astype('float32'))

        test_data.append(data[i][test_inx].reshape([test_inx.shape[0], 1, dim, -1], order='F').astype('float32'))
        test_labels.append((labels[i][test_inx]).astype('float32'))
    data = {'train': train_data, 'valid': valid_data, 'test': test_data}
    labels = {'train': train_labels, 'valid': valid_labels, 'test': test_labels}
    return data, labels

def get_loader(dataset, batch_size):
    wv_matrix, MAP = None, None
    if dataset in data_names:
        data, labels, wv_matrix = get_ImgTxt(dataset=dataset)
        MAP = 0
    elif dataset == 'reuters':
        data, labels = get_Reuters()
        MAP = 0
    elif dataset == 'nMSAD':
        data, labels = get_nMSAD()
        MAP = 0
    
    shuffle = {'train': True, 'valid': False, 'test': False}
    dataset = {x: CustomDataSet(data=data[x], labels=labels[x]) for x in ['train', 'valid', 'test']}
    dataloader = {x: DataLoaderX(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=3) for x in ['train', 'valid', 'test']}

    input_data_par = {}
    input_data_par['data_test'] = data['test']
    input_data_par['label_test'] = labels['test']
    input_data_par['data_train'] = data['train']
    input_data_par['label_train'] = labels['train']
    input_data_par['data_valid'] = data['valid']
    input_data_par['label_valid'] = labels['valid']
    input_data_par['in_dims'] = [d.shape[1::] for d in data['train']]
    input_data_par['wv_matrix'] = wv_matrix
    return dataloader, input_data_par
