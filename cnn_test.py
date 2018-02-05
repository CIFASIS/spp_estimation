#!/usr/bin/env python2.7

import os
import argparse
import pickle
from six.moves import urllib
import tarfile
from time import time
import numpy as np
from numpy.random import RandomState
import theano
import theano.tensor as T
import lasagne
from lasagne.updates import adam
from lasagne.init import Normal

from lasagne_utils import load_network_values, save_network_values

from numpy_dataset import Numpy_dataset
from data import get_batch_indexes, scale_data, Dataset, floatX
from augmentation import Augmentation, zoom_and_crop
from cnn import build_classifier
from sklearn.model_selection import GroupKFold

from lasagne.objectives import categorical_crossentropy
from lasagne.regularization import regularize_layer_params, l2

from sklearn.metrics import confusion_matrix

def build_predict_function(classifier, X=T.tensor4()):
    test_clsX = lasagne.layers.get_output(classifier, X, deterministic=True)
    return theano.function(inputs=[X],outputs=test_clsX,updates=None)

def mean_acc_by_fold(y_true, y_pred, groups):

    n_groups = max(groups) + 1
    n_splits = n_groups if n_groups <= 10 else 5
    gkf = GroupKFold(n_splits=n_splits)
    acc = []
    for _, group_indexes in gkf.split(y_pred, y_true, groups=groups):
        comp = y_true[group_indexes]==y_pred[group_indexes]
        acc.append(comp.mean())
    acc = np.asarray(acc)
    return acc.mean(), acc.std()




def train_model(params, data, test_groups, samples_dir='samples', best_model_dir='best_model', grid_size=8, verbosity=0):
    batch_size = params["batch_size"]
    n_iter = params["n_iter"]
    lr = 10.**params["log10lr"]
    iter_save = params["iter_save"]
    model_width = params["model_width"]
    seed = params["seed"]
    wd = np.asscalar(floatX(10.**params["log10wd"]))
    rotation_range = params["rotation_range"]
    width_shift_range = params["width_shift_range"]
    height_shift_range = params["height_shift_range"]
    shear_range = params["shear_range"]
    zoom_range = (params["zoom_range_center"]-params["zoom_range_range"]*0.5,
                  params["zoom_range_center"]+params["zoom_range_range"]*0.5)
    random_curves_strength = params["random_curves_strength"]
    n_layers_per_block = params["n_layers_per_block"]
    n_blocks = params["n_blocks"]


    test_size = data.X_test.shape[0]
    valid_size = data.X_valid.shape[0]
    train_size = data.X_train.shape[0]
    nchannels = data.X_train.shape[1]
    img_shape = data.X_train.shape[2:4]
    nclasses = data.nclasses
    class_labels = None  # TODO
    y_test_pred = np.zeros((test_size + batch_size,))
    assert(test_size==data.y_test.shape[0])

    # MODELS
    strides = ([1] * (n_layers_per_block - 1) + [2]) * n_blocks
    classifier = build_classifier(nclasses=nclasses,
                                  img_size=img_shape,
                                  nchannels=nchannels,
                                  ndf=model_width,
                                  global_pool=True,
                                  strides=strides)

    it_best = 0
    best_acc = 0.0
    if n_iter>0:

        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir)

        augm = Augmentation(rotation_range=rotation_range,  # In degrees
                            width_shift_range=width_shift_range,
                            height_shift_range=height_shift_range,
                            horizontal_flip=True,
                            shear_range=shear_range,  # In radians
                            zoom_range=zoom_range,  # >1 zoom out; <1 zoom in
                            channel_shift_range=0.0,  # 0-255
                            fill_mode='constant',  # 'nearest',
                            random_curves_strength=random_curves_strength,
                            seed=seed)

        if verbosity>0:
            print  data.X_train.shape, img_shape

        # SYMBOLIC INPUTS
        X = T.tensor4()
        y = T.matrix()


        clsX = lasagne.layers.get_output(classifier, X)

        # LOSS FUNCTIONS
        weight_decay = regularize_layer_params(classifier, l2)
        cls_loss = categorical_crossentropy(clsX,y).mean() +  wd * weight_decay

        # PARAMS
        cls_params = lasagne.layers.get_all_params(classifier, trainable=True)

        # UPDATES
        cls_updates = adam(cls_loss, cls_params, learning_rate= floatX(lr), beta1=floatX(0.9), beta2=floatX(0.999))

        # TRAINING FUNCTIONS
        if verbosity>0:
            print 'COMPILING TRAINING FUNCTIONS'
        t = time()
        train_cls = theano.function([X, y], cls_loss, updates=cls_updates)
        if verbosity>0:
            print '%.2f seconds to compile theano functions' % (time() - t)

        # MONITOR
        if verbosity > 0:
            print 'COMPILING MONITOR FUNCTIONS'
        t = time()
        predict = build_predict_function(classifier, X)
        if verbosity > 0:
            print '%.2f seconds to compile theano functions' % (time() - t)

        if verbosity > 0:
            print "starting training"
            with open(best_model_dir + '/accuracies.log', 'w') as f:
                f.write('# iter data_seen epoch cls_loss train_acc valid_acc')
                f.write('\n')
            with open(best_model_dir + '/best_acc.log', 'w') as f:
                f.write('# iter data_seen epoch valid_acc test_acc test_acc_mean test_acc_std')
                f.write('\n')

        n_epochs =  n_iter* batch_size/train_size

        last_it = 0
        t = time()
        for it in xrange(0, n_iter):
            epoch = it* batch_size/train_size

            X_batch, y_batch = data.get_train_batch(it, batch_size)
            X_batch = augm.random_transform(X_batch)
            X_batch = data.scale_data(X_batch)
            cls_loss_value = train_cls(X_batch, y_batch)

            if (it %  iter_save == 0) or (it % 10 == 0 and it <  iter_save):
                y_pred = np.argmax(predict(X_batch), axis=1)
                y_true = np.argmax(y_batch, axis=1)
                train_acc = (y_pred == y_true).mean()

                y_pred = np.asarray([])
                y_true = np.asarray([])
                for valit in range(valid_size/ batch_size):
                    X_valid, y_valid = data.get_valid_batch(valit, batch_size)
                    X_valid = data.scale_data(X_valid)
                    y_pred = np.append(y_pred, np.argmax(predict(X_valid), axis=1))
                    y_true = np.append(y_true, np.argmax(y_valid, axis=1))
                valid_acc = (y_pred == y_true).mean()
                valid_cm = confusion_matrix(y_true,y_pred,class_labels)

                if verbosity>0:
                    print train_acc, valid_acc

                if best_acc<valid_acc:
                    best_acc = valid_acc
                    it_best = it + 1
                    if verbosity>2:
                        save_network_values(classifier, os.path.join(best_model_dir, 'classifier.npz'))
                        pickle.dump(params,
                                    open("%s/ p" % best_model_dir, "wb"))

                    for testit in range(test_size / batch_size + 1):
                        X_test, y_test = data.get_test_batch(testit,  batch_size)
                        X_test = data.scale_data(X_test)
                        pred = predict(X_test)
                        y_test_pred[testit * batch_size:(testit + 1) * batch_size] = np.argmax(pred, axis=1)


                    y_p = y_test_pred[:test_size]
                    y_t = data.y_test
                    test_acc_mean, test_acc_std = mean_acc_by_fold(y_p, y_t, test_groups)
                    test_acc = (y_p == y_t).mean()
                    test_cm = confusion_matrix(y_t, y_p, class_labels)

                    with open(best_model_dir + '/best_acc.log', 'a') as f:
                        np.savetxt(f, [[it + 1, (it + 1) *  batch_size, epoch,
                                        valid_acc, test_acc, test_acc_mean, test_acc_std]], fmt='%1.3e')
                    if verbosity>0:
                        print "Best valid accuracy reached: %2.2f%%  "%(valid_acc*100)
                        print "Test Acc.: %2.2f%%"%(test_acc*100)
                        print "Mean Test Acc.: %1.3f +/- %1.3f"%(test_acc_mean,test_acc_std)
                        print "valid CM\n", valid_cm
                        print "test CM\n", test_cm

                with open(best_model_dir + '/accuracies.log', 'a') as f:
                    np.savetxt(f, [[it+1, (it+1)* batch_size, epoch,
                                  cls_loss_value, train_acc, valid_acc]], fmt='%1.3e')

                if verbosity>0:
                    t2 = time()-t
                    t += t2
                    horas = t2/(1+it-last_it)/3600.*10000
                    print "iter:%d/%d; epoch:%d;    %4.2f hours for 10000 iterations"%(it+1, n_iter,epoch,horas)
                    last_it = it+1


        print "End train\n"

    load_network_values(classifier, os.path.join(best_model_dir, 'classifier.npz'))
    best_predict = build_predict_function(classifier)

    return best_acc, it_best, best_predict


def mk_classification_images(data, which_set, file, predict, samples_dir, batch_size=32):

    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    if which_set=='train':
        X = data.X_train
    elif which_set=='valid':
        X = data.X_valid
    elif which_set=='test':
        X = data.X_test
    size = X.shape[0]
    y_pred = np.zeros((size+batch_size,))
    y_true = np.zeros((size+batch_size,))
    y_maxprob = np.zeros((size+batch_size,))

    for it in range(size / batch_size + 1):
        if which_set == 'train':
            X_batch, y_batch = data.get_train_batch(it, batch_size)
        elif which_set == 'valid':
            X_batch, y_batch = data.get_valid_batch(it, batch_size)
        elif which_set == 'test':
            X_batch, y_batch = data.get_test_batch(it, batch_size)
        X_batch = data.scale_data(X_batch)

        pred = predict(X_batch)
        y_pred[it*batch_size:(it+1)*batch_size] = np.argmax(pred, axis=1)
        y_true[it*batch_size:(it+1)*batch_size] = np.argmax(y_batch, axis=1)
        y_maxprob[it*batch_size:(it+1)*batch_size] = np.max(pred, axis=1)
    y_pred_arr = y_pred[:size]
    y_true_arr = y_true[:size]
    y_maxprob_arr = y_maxprob[:size]


def eval_params(params,data,test_data,samples_dir='samples',best_model_dir='best_model',grid_size=8, verbosity=2):

    n_groups = max(data.groups) + 1
    n_splits = n_groups if n_groups<=10 else 5
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    for train_indexes, valid_indexes in gkf.split(data.X, data.y, groups=data.groups):
        np.random.shuffle(train_indexes)
        dataset = Dataset(X_train = data.X[train_indexes],
                          y_train = data.y[train_indexes],
                          X_valid = data.X[valid_indexes],
                          y_valid = data.y[valid_indexes],
                          X_test = test_data.X,
                          y_test = test_data.y, nclasses = 3)


        acc, n_iter, _ = train_model(params,dataset,test_data.groups,samples_dir,best_model_dir,grid_size,verbosity=verbosity)
        print acc, n_iter
        accuracies.append(acc)

    accuracies = np.asarray(accuracies)
    print "Mean Accuracy: (%2.2f +/- %2.2f)"%(accuracies.mean()*100, accuracies.std()*100)
    return accuracies

def main(args):
    # print args

    # DATASET
    DATA_URL  = 'http://www.cifasis-conicet.gov.ar/uzal/dataset/soybean_pods.tar.gz'
    filepath = os.path.join(args.datapath,'soybean_pods.tar.gz')
    datapath = os.path.join(args.datapath, 'soybean_pods/')
    if not os.path.exists(datapath):
        #os.makedirs(args.datapath)
        print('Downloading soybean_pods.tar.gz')
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded soybean_pods.tar.gz', statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(args.datapath)

    data_train_dir = os.path.join(datapath,"Season1")
    data_test_dir = os.path.join(datapath,"Season2")
    data = Numpy_dataset(data_train_dir,'train.npz')
    test_data = Numpy_dataset(data_test_dir,'train.npz')

    params = vars(args)

    n_groups = max(data.groups) + 1
    n_splits = n_groups if n_groups <= 10 else 5
    gkf = GroupKFold(n_splits=n_splits)
    fold = 1
    for train_indexes, valid_indexes in gkf.split(data.X, data.y, groups=data.groups):
        np.random.shuffle(train_indexes)
        dataset = Dataset(X_train=data.X[train_indexes],
                          y_train=data.y[train_indexes],
                          X_valid=data.X[valid_indexes],
                          y_valid=data.y[valid_indexes],
                          X_test=test_data.X,
                          y_test=test_data.y,nclasses=3)


        acc, n_iter, predict = train_model(params,dataset,test_data.groups,
                                  samples_dir='samples_fold%d'%fold,
                                  best_model_dir='best_model_fold%d'%fold,
                                  grid_size=8,verbosity=10)
        print acc, n_iter

        fold = fold + 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/tmp')
    parser.add_argument("--n_iter", type=int, default=6000)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--iter_save", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument('--n_layers_per_block', type=int, default=4)
    parser.add_argument('--n_blocks', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_width', type=int, default=16)
    parser.add_argument('--log10wd', type=float, default=-1.099842)
    parser.add_argument('--log10lr', type=float, default=-2.42706)
    parser.add_argument('--rotation_range', type=int, default=20)
    parser.add_argument('--width_shift_range', type=float, default=0.028)
    parser.add_argument('--height_shift_range', type=float, default=0.016)
    parser.add_argument('--shear_range', type=float, default=0.14)  # In radians
    parser.add_argument('--zoom_range_center', type=float, default=0.97)
    parser.add_argument('--zoom_range_range', type=float, default=0.18)
    parser.add_argument('--random_curves_strength', type=float, default=0.58)
    parser.add_argument('--occlusion', type=int, default=5)

    args = parser.parse_args()

    main(args)
