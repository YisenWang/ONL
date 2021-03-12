from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np
import keras.backend as K
import tensorflow as tf
import argparse
from util import get_data, get_model
import os
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda

from sklearn.neighbors import LocalOutlierFactor
import pdb
import time

PATH_DATA = './data'

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = tf.cast(y_true, tf.float32)
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def weight_ce(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

def create_train_pairs(x, y, weighted_y_train, outlier_indices, noise_indices, true_indices):
    pairs = []
    labels = []
    y1 = []
    y2 = []
    weight = []
    non_outliers = {}

    y = np_utils.to_categorical(y, 10)

    for d in range(10):
        non_outliers[d] = np.setdiff1d(noise_indices[d], outlier_indices[d])

    n = min([len(non_outliers[d]) for d in range(10)]) -1

    for d in range(10):
        for i in range(n):
            z1, z2 = non_outliers[d][i], non_outliers[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            y1 += [y[z1]]
            y2 += [y[z2]]
            weight += [weighted_y_train[z2]]

            z1, z2 = non_outliers[d][i], outlier_indices[d][i%len(outlier_indices[d])]
            pairs += [[x[z1], x[z2]]]
            y1 += [y[z1]]
            y2 += [y[z2]]
            weight += [weighted_y_train[z2]]

            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = non_outliers[d][i], non_outliers[dn][i]
            pairs += [[x[z1], x[z2]]]
            y1 += [y[z1]]
            y2 += [y[z2]]
            weight += [weighted_y_train[z2]]

            labels += [1, 0, 0]

    return np.array(pairs), np.array(labels), np.array(y1), np.array(y2), np.array(weight)


def create_test_pairs(x, y, digit_indices):
    random.seed(0)
    pairs = []
    labels = []
    y1 = []
    y2 = []

    y = np_utils.to_categorical(y, 10)

    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            y1 += [y[z1]]
            y2 += [y[z2]]

            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            y1 += [y[z1]]
            y2 += [y[z2]]

            labels += [1, 0]
    return np.array(pairs), np.array(labels), np.array(y1), np.array(y2)

def get_deep_representations(model, X, batch_size=100):
    K.set_learning_phase(0)
    output_dim = model.layers[-4].get_output_at(0).shape[-1].value
    get_encoding = K.function(
        [model.layers[0].get_input_at(0), K.learning_phase()],
        [model.layers[-4].get_output_at(0)]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[1]

    return output


def train(dataset='cifar10', batch_size=100, epochs=50, m=0):
    print('Data set: %s' % dataset)

    noiserate = 0.4
    if m == 0:
        X_train, Y_train, X_test, Y_test = get_data(dataset, noiserate)
    else:
        X_train = np.load('./data/X_train.npy', allow_pickle=True)
        Y_train = np.load('./data/y_train.npy', allow_pickle=True)
        X_test = np.load('./data/X_test.npy', allow_pickle=True)
        Y_test = np.load('./data/y_test.npy', allow_pickle=True)

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)

    Y_true = Y_train[0:int(Y_train.shape[0] * (1 - noiserate))]
    true_indices = [np.where(Y_true == i)[0] for i in range(10)]
    noise_indices = [np.where(Y_train == i)[0] for i in range(10)]

    np.save('./data/true_indices', true_indices)
    np.save('./data/noise_indices', noise_indices)

    if m == 0:
        tr_pairs, tr_y, y1, y2 = create_test_pairs(X_train, Y_train, noise_indices)

    if m >= 1:
        outlier_indices = np.load('./data/%s_%d_train_outlier_index.npy' % (dataset, m - 1), allow_pickle=True)
        weighted_y_train = np.load('./data/%s_%d_train_weighted_y_train.npy' % (dataset, m - 1), allow_pickle=True)
        tr_pairs, tr_y, y1, y2, weight = create_train_pairs(X_train, Y_train, weighted_y_train, outlier_indices, noise_indices,
                                                    true_indices)

    base_network = get_model('cifar10')

    input_a = Input(shape=(32, 32, 3), name='input_a')
    input_b = Input(shape=(32, 32, 3), name='input_b')

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape, name='contrastive')([processed_a, processed_b])

    output1 = Dense(10, activation='softmax', name='output1')(processed_a)
    output2 = Dense(10, activation='softmax', name='output2')(processed_b)

    model = Model(inputs=[input_a, input_b], outputs=[distance, output1, output2])

    if m == 0:
        model.compile(
            optimizer='adam',
            loss={'contrastive': contrastive_loss, 'output1': weight_ce, 'output2': weight_ce},
            loss_weights={'contrastive': 1., 'output1': 1., 'output2': 1.}
        )
        training_start_time = time.time()

        model.fit(
            {'input_a': tr_pairs[:, 0], 'input_b': tr_pairs[:, 1]}, {'contrastive': tr_y, 'output1': y1, 'output2': y2},
            epochs=5,
            batch_size=batch_size,
            verbose=1
        )
    else:
        model.load_weights('./data/model_%s_%d_outlier_detection_training.h5' % (dataset, m - 1))
        model.compile(
            optimizer='adam',
            loss={'contrastive': contrastive_loss, 'output1': weight_ce, 'output2': weight_ce},
            loss_weights={'contrastive': 1., 'output1': 1., 'output2': 1.}
        )
        training_start_time = time.time()

        model.fit(
            {'input_a': tr_pairs[:, 0], 'input_b': tr_pairs[:, 1]}, {'contrastive': tr_y, 'output1': y1, 'output2': y2},
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            sample_weight={'contrastive': np.ones(y1.shape[0]), 'output1': np.ones(y1.shape[0]), 'output2': weight}
        )

    print('Iteration %d training time: %0.2f' % (m + 1, time.time() - training_start_time))

    _, y_train_pred, _ = model.predict([X_train, X_train])
    y_train_true = np_utils.to_categorical(Y_train, 10)
    tr_acc = np.equal(np.argmax(y_train_true, axis=1), np.argmax(y_train_pred, axis=1)).mean()

    _, y_test_pred, _ = model.predict([X_test, X_test])
    y_test_true = np_utils.to_categorical(Y_test, 10)
    te_acc = np.equal(np.argmax(y_test_true, axis=1), np.argmax(y_test_pred, axis=1)).mean()

    print('*****get representation*****')
    representation_layer_model = Model(inputs=input_a,
                                       outputs=processed_a)
    X_train_features = representation_layer_model.predict(X_train)

    print('X_train_features', X_train_features.shape)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    model.save('./data/model_%s_%d_outlier_detection_training.h5' % (dataset, m))

    return X_train, y_train_true, noise_indices, X_train_features


def get_outlier_factor(X_train_features, y_train_true, class_indices, cumulative_score, m):
    clf = LocalOutlierFactor(n_neighbors=int(2000), metric='cosine', contamination=0.4)
    scores = {}
    for i in range(10):
        clf.fit(X_train_features[class_indices[i]])
        scores[i] = - 1.0 / clf.negative_outlier_factor_
        if m == 0:
            cumulative_score[i] = scores[i]
        else:
            cumulative_score[i] += scores[i]
    outlier = [np.where(cumulative_score[i] <= np.percentile(cumulative_score[i], 50))[0] for i in range(10)]
    outlier_index = [class_indices[i][outlier[i]] for i in range(10)]
    weight = [cumulative_score[i][outlier[i]] / (m+1) for i in range(10)]

    for i in range(10):
        for index, j in enumerate(outlier_index[i]):
            y_train_true[j][i] = weight[i][index]
    weighted_y_train = np.max(y_train_true, axis = 1)/10.0

    return outlier_index, weighted_y_train


def main(args):
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar10', 'svhn']:
            train(dataset, args.batch_size, args.epochs, args.clean_training)
    else:
        cumulative_score = {}
        for m in range(10):
            X_train, y_train_true, class_indices, X_train_features = train(args.dataset, args.batch_size, args.epochs,
                                                                           m=m)
            outlier_index, weighted_y_train = get_outlier_factor(X_train_features, y_train_true, class_indices,
                                                                 cumulative_score, m)
            outlier_file = os.path.join(PATH_DATA, "%s_%d_train_outlier_index.npy" % (args.dataset, m))
            np.save(outlier_file, outlier_index)
            weighted_y_train_file = os.path.join(PATH_DATA, "%s_%d_train_weighted_y_train.npy" % (args.dataset, m))
            np.save(weighted_y_train_file, weighted_y_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar10', 'svhn' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )

    args = parser.parse_args(['-d', 'cifar10-cifar100', '-e', '10', '-b', '100'])
    main(args)