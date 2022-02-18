import os 
import glob
import numpy as np
import six
import pickle as cPickle
import tensorflow as tf

from imageio import imread

import scipy.io as sio



def one_hot_encoded(class_numbers, num_classes):
    return np.eye(num_classes, dtype=float)[class_numbers]


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
        
class SubModel():
    def __init__(self,patch_size_x,patch_size_y,path):
        self.num_classes = 1
        print('SubModel called')
        W_name = os.path.join(path, 'W_l.npy')
        yw_name = os.path.join(path, 'yw_l.npy')
        self.imgs = np.load(W_name) 
        self.test_imgs = np.load(W_name)
        self.labels = np.load(yw_name)
        self.test_labels = np.load(yw_name)

        self.labels = one_hot_encoded(self.labels, 1)        # defined in bgan_util.py
        self.test_labels = one_hot_encoded(self.test_labels, 1) 
        self.x_dim = [patch_size_x,patch_size_y,1]
#        self.x_dim = [80,80,1]

        self.dataset_size = self.imgs.shape[0]

    @staticmethod
    def get_batch(batch_size, x, y): 
        """Returns a batch from the given arrays.
        """
        idx = np.random.choice(range(x.shape[0]), size=(batch_size,), replace=False)
        return x[idx], y[idx]

    def next_batch(self, batch_size, class_id=None):
        return self.get_batch(batch_size, self.imgs, self.labels)

    def test_batch(self, batch_size):
        return self.get_batch(batch_size, self.test_imgs, self.test_labels)
    
