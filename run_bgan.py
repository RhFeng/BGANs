#!/usr/bin/env python

import os
import sys
import argparse
import json
import datetime
import time

import numpy as np
from math import ceil

from PIL import Image

import tensorflow as tf


from bgan_util import AttributeDict
from bgan_util import SubModel
from bgan import BDCGAN


def get_session():

    print("Creating new session")

    from tensorflow.python.framework import ops
    ops.reset_default_graph()
    _SESSION = tf.compat.v1.InteractiveSession()


    return _SESSION


def b_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size
    dataset_size = dataset.dataset_size

    session = get_session()

    
    dcgan = BDCGAN(x_dim, z_dim, dataset_size, batch_size=batch_size,
                   J=args.J, J_d=args.J_d, M=args.M,
                   num_layers=args.num_layers,
                   lr=args.lr, optimizer=args.optimizer, gf_dim=args.gf_dim, 
                   df_dim=args.df_dim,
                   ml=(args.ml and args.J==1 and args.M==1 and args.J_d==1))
    
    print("Starting session")
    session.run(tf.compat.v1.global_variables_initializer())

    print("Starting training loop")
        
    num_train_iter = args.train_iter

    optimizer_dict = {"disc": dcgan.d_optims_adam,
                      "gen": dcgan.g_optims_adam}

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay
    num_disc = args.J_d
    
    results = {}
    
    for train_iter in range(num_train_iter):

        if train_iter == 5000:
            print("Switching to user-specified optimizer")
            optimizer_dict = {"disc": dcgan.d_optims_adam,
                              "gen": dcgan.g_optims_adam}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))
                                                    
        image_batch, _ = dataset.next_batch(batch_size, class_id=None)       

        ### compute disc losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        disc_info = session.run(optimizer_dict["disc"] + dcgan.d_losses, 
                                feed_dict={dcgan.inputs: image_batch,
                                           dcgan.z: batch_z,
                                           dcgan.d_learning_rate: learning_rate})


        
        d_losses = [d_ for d_ in disc_info if d_ is not None]

        ### compute generative losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        gen_info = session.run(optimizer_dict["gen"] + dcgan.g_losses,
                               feed_dict={dcgan.z: batch_z,
                                          dcgan.inputs: image_batch,
                                          dcgan.g_learning_rate: learning_rate})
        
        g_losses = [g_ for g_ in gen_info if g_ is not None]
        
        print("Iter %i" % train_iter)
        print("Gen losses = %s" % (", ".join(["%.2f" % gl for gl in g_losses])))
        print("Disc losses = %s" % (", ".join(["%.2f" % dl for dl in d_losses])))
        
        results[train_iter] = {"disc_losses": map(float, d_losses),
                       "gen_losses": map(float, g_losses),
                       "timestamp": time.time()}
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")
    
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    
    parser.add_argument('--gf_dim',
                        type=int,
                        default=64,
                        help='num of gen features')
    
    parser.add_argument('--df_dim',
                        type=int,
                        default=96,
                        help='num of disc features')
    
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to where the datasets live')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="minibatch size")

    parser.add_argument('--prior_std',
                        type=float,
                        default=1.0,
                        help="NN weight prior std.")

    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help="number of layers for G and D nets")

    parser.add_argument('--num_gen',
                        type=int,
                        dest="J",
                        default=1,
                        help="number of samples of z/generators")

    parser.add_argument('--num_disc',
                        type=int,
                        dest="J_d",
                        default=1,
                        help="number of discrimitor weight samples")

    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=1,
                        help="number of MCMC NN weight samples per z")

    parser.add_argument('--train_iter',
                        type=int,
                        default=50000,
                        help="number of training iterations")

    parser.add_argument('--ml',
                        action="store_true",
                        help="if specified, disable bayesian things")

    parser.add_argument('--random_seed',
                        type=int,
                        default=2222,
                        help="random seed")
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate")
                        
    parser.add_argument('--lr_decay',
                        type=float,
                        default=3.0,
                        help="learning rate")                    

    parser.add_argument('--optimizer',
                        type=str,
                        default="adam",
                        help="optimizer --- 'adam' or 'sgd'")
                        
    parser.add_argument('--patch_size_x',
                        type=int,
                        default=28,
                        help="patch size")
    
    parser.add_argument('--patch_size_y',
                        type=int,
                        default=28,
                        help="patch size")


    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir)

    # set seeds
    np.random.seed(args.random_seed)
    
    dataset = SubModel(args.patch_size_x,args.patch_size_y,args.data_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    print(tf.test.gpu_device_name())

    ### main call
    b_dcgan(dataset, args)
