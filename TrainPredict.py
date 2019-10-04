import json
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import load_npz, dok_matrix
import numpy as np
import pickle
from keras.models import Model, load_model
from keras.layers import Dense, GaussianNoise, Input, Embedding, Lambda
from keras.utils import to_categorical
from tqdm import tqdm
import pandas as pd
import os
from sklearn.utils  import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import keras.backend as K
from functools import reduce
import gc 
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse, BytesLimit
import tensorflow as tf


def print_bytes():
    b_i_u = BytesInUse()
    b_l = BytesLimit()
    with tf.Session() as sess:
        print('{} Bytes usados de {}'.format(sess.run(b_i_u), sess.run(b_l)))
    pass

def to_categorical_sparse(y, classes=None):
    if classes is None:
        classes = np.unique(y).shape[0]
    res = dok_matrix((y.shape[0], classes))
    res[range(y.shape[0]), y] = 1
    return res.tocsr()


def model_predict(lng, n_models=10, test_size=0.01, model_name='base'):
    print('Loading...')
    base_dir = 'separated_seq' + os.sep
    
    x_full = json.load(open(base_dir + 'x_{}.json'.format(lng), 'r', encoding='utf-8'))
    x = x_full['x_{}'.format(lng)]
    x_test = x_full['x_test_{}'.format(lng)]
    x_len = max(map(len, x))
    x_voc = len(pickle.load(open(base_dir + 'words_{}.p'.format(lng), 'rb')))
    y_w_idx = np.load(base_dir + 'y_w_idx_{}.npz'.format(lng))
    y = y_w_idx['y_{}'.format(lng)]
    w = np.where(y_w_idx['w_{}'.format(lng)] > 0.6, 1, 0.2) * compute_sample_weight('balanced', y)
    idx = y_w_idx['idx_{}'.format(lng)]
    classes = pickle.load(open(base_dir + 'classes_{}.p'.format(lng), 'rb'))
    print('Padding...')
    x = pad_sequences(x, maxlen=x_len)
    x_test = pad_sequences(x_test, maxlen=x_len)
    
    base_dir += 'model_{}_{}'.format(model_name, lng)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    model_path = base_dir + os.sep + 'model_{}.h5'
    val_path = base_dir + os.sep + 'val_{}.npz'

    print('Defining model')
    for m in range(n_models):
        if os.path.exists(model_path.format(m)) and os.path.exists(val_path.format(m)):
            continue
        print('Training model {} of {} for {}'.format(m, n_models, lng))
        print_bytes()
        x_train, x_val, y_train, y_val, w_train, _, _, id_val = train_test_split(x, y, w, np.asarray(list(range(x.shape[0]))), test_size=test_size, stratify=y)

        i = Input((x.shape[1],))
        d = Embedding(x_voc + 1, 400)(i)
        d = Lambda(lambda x: K.sum(x, axis=1))(d)
        d = Dense(len(classes), activation='softmax')(d)
        model = Model(i, d)
        model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        model.fit(x_train, to_categorical_sparse(y_train, len(classes)), epochs=1, batch_size=5000, sample_weight=w_train)
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr)/10)
        model.fit(x_train, to_categorical_sparse(y_train, len(classes)), epochs=1, batch_size=5000, sample_weight=w_train)

        y_pred_val = np.argmax(model.predict(x_val, verbose=1, batch_size=10000), axis=1)
        model.save(model_path.format(m))
        np.savez(val_path.format(m), y_true=y_val, y_pred=y_pred_val, x_val=x_val, id_val=id_val)
        print_bytes()
        del i
        del d
        del model
        del x_train
        del y_train
        del x_val
        del y_val
        del y_pred_val
        gc.collect()
        K.clear_session()
        gc.collect()
    del x
    del y
    del w
    y_pred = np.zeros((x_test.shape[0], len(classes)))
    for m in range(n_models):
        model = load_model(model_path.format(m))
        val_data = np.load(val_path.format(m))
        recall = recall_score(val_data['y_true'], val_data['y_pred'], labels=range(len(classes)), average=None)
        recall = np.where(recall == 0, 1, recall)
        y_pred += recall * model.predict(x_test, verbose=1, batch_size=10000)
        del model
        del val_data
        gc.collect()
        K.clear_session()
        gc.collect()

    y_pred = np.argmax(y_pred, axis=1)
    inverse_classes = [None] * len(classes)
    for k, v in classes.items():
        inverse_classes[v] = k

    res = [inverse_classes[c] for c in tqdm(y_pred)]
    return res, idx


def make_submition(res_sp, idx_sp, res_pt, idx_pt, model_name='base'):
    base_dir = 'separated_seq'+os.sep
    res = [None] * (len(res_sp) + len(res_pt))
    i_sp = 0
    i_pt = 0
    for i in range(len(res)):
        if i_sp < len(res_sp) and idx_sp[i_sp] == i:
            res[i] = [i, res_sp[i_sp]]
            i_sp += 1
        else:
            res[i] = [i, res_pt[i_pt]]
            i_pt += 1
    ds_test_a = pd.DataFrame(data=np.asarray(res), columns=['id', 'category'])
    ds_test_a.to_csv(base_dir + 'submit_{}.csv'.format(model_name), index=False)
    pass


def train_predict():
    model_name = 'multi_model_per_class_400emb_lr_sum_softmax_0_20_reem1'
    res_sp, idx_sp = model_predict('sp', model_name=model_name)
    res_pt, idx_pt = model_predict('pt', model_name=model_name)
    make_submition(res_sp, idx_sp, res_pt, idx_pt, model_name=model_name)
    pass


if __name__ == '__main__':
    train_predict()
