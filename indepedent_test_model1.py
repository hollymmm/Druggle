# 添加了输出中间层的值
#
import os
import pickle
import random
import warnings

import matplotlib.pyplot as plt
import shap
import tensorflow

# tensorflow.compat.v1.disable_v2_behavior()
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import BatchNormalization, Bidirectional, LSTM, GRU
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizer_v2.adam import Adam

from sklearn.metrics import *
import numpy as np
# Initial settings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from numpy.random import seed
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from numpy.random import seed
from tensorflow.keras import backend as K
import time

import tensorflow as tf


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tensorflow.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        print("Training on GPU...")
        for gpu in tf.config.experimental.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)


def get_middle_output(model):
    middle = Model(inputs=model.input, outputs=model.get_layer('o1').output)
    print("1")


set_tf_device('gpu')  # 'cpu' or 'gpu'

setup_seed(2022);  # 设置随机种子


def eva(yy_pred, true_values):
    y_pred = yy_pred
    y_scores = np.array(y_pred)
    AUC = roc_auc_score(true_values, y_scores)
    y_labels = []
    y_scores = y_scores.reshape((len(y_scores), -1))
    for i in range(len(y_scores)):
        if (y_scores[i] >= 0.5):
            y_labels.append(1)
        else:
            y_labels.append(0)
    print("pre_label:")
    print(y_labels)
    acc = accuracy_score(true_values, y_labels)
    se = recall_score(true_values, y_labels)
    sp = recall_score(true_values, y_labels, pos_label=0)
    mcc = matthews_corrcoef(true_values, y_labels)
    precision = precision_score(true_values, y_labels)
    f1 = f1_score(true_values, y_labels)
    precision2, recall, thresholds = precision_recall_curve(true_values, y_labels)
    AUPRC = auc(recall, precision2)
    return acc, se, sp, mcc, AUC, precision, f1, AUPRC


def mlp(input_shape):
    model = Sequential()
    model.add(Dense(input_shape, input_dim=input_shape, name='dense1'))
    model.add(Dense(64, name='dense3'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def bilstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True), name='o1'))
    model.add(Flatten())
    model.add(Dense(16, name='o2'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def bigru(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Bidirectional(GRU(units=64, batch_input_shape=(None, 1, input_shape), return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(16,name='bigru_dense'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run(feature_name, model_name):
    print(feature_name)
    print(model_name)


    ##################################1.读入特征#################################################

    path1 = './data/'+ feature_name + '/train_' + feature_name + '.csv'
    train = pd.read_csv(path1, header=0, index_col=0)
    save_column = train.columns
    save_column=['Tape_90','ProtT5_3','ProtT5_10','ProtBERT-BFD_0','TAPE_13',
                 'ASDC_55','ProtT5_59','ProtT5_62','ProtBERT-BFD_62','Tape_104','TAPE_123',
                 'ASDC_129','ProtBERT-BFD_229','ProtT5_95','ProtT5_88','ProtT5_29',
                 'ProtT5_68','ProtT5_15','ProtT5_30','ASDC_48'
                 ]
    # 'tape_avg_123', 'ASDC_129', 'prot_bert_bfd_229', 't5_95', 't5_88',
    # 't5_29', 't5_68', 't5_15', 't5_30', 'ASDC_48'],
    path1 = './data/'+ feature_name + '/test_' + feature_name + '.csv'
    test = pd.read_csv(path1, header=0, index_col=0)
    train = np.array(train)
    test = np.array(test)

    pos_num = 1224
    neg_num = 1319
    train_label = np.array([1] * pos_num + [0] * neg_num)
    pos_num = 224
    neg_num = 237
    test_label = np.array([1] * pos_num + [0] * neg_num)

    ###############################2.训练####################################################
    isExists = os.path.exists('./ind_save/' + feature_name + '_' + model_name)
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs('./ind_save/' + feature_name + '_' + model_name)
        print("%s 目录创建成功")
    if (model_name == 'svm'):
        model = SVC(probability=True).fit(train, train_label)
        y_pred = model.predict_proba(test)[:, 1]
    if (model_name == 'mlp'):
        train, val, train_label, val_label = train_test_split(train, train_label, test_size=0.2, random_state=123)
        model_save_path = './model/' + feature_name + "_" + model_name + '.h5'
        model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        model = mlp(train.shape[1])
        model.fit(train, train_label, batch_size=64, epochs=40, shuffle=True, validation_data=(val, val_label)
                  , callbacks=[model_check, reduct_L_rate])
        model = load_model(model_save_path)
        y_pred = model.predict(test)
    if (model_name == 'bilstm'):
        train = train[:, np.newaxis]
        test = test[:, np.newaxis]
        train, val, train_label, val_label = train_test_split(train, train_label, test_size=0.2, random_state=123)
        # 输出划分后的label
        # output_label=pd.DataFrame(train_label)
        # output_label.to_csv("train_label.csv")

        model_save_path = './model/' + feature_name + "_" + model_name + '.h5'
        model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        model = bilstm(train.shape[1])
        model.fit(train, train_label, batch_size=64, epochs=128, shuffle=True, validation_data=(val, val_label)
                  , callbacks=[model_check, reduct_L_rate])

        # 保存输出
        # middle_output = pd.DataFrame(middle_output)
        # middle_output.to_csv("./output/middle_output/" + feature_name + '_' + 'bilstm_o1' + ".csv")

        # model_middle = Model(inputs=model.input, outputs=model.get_layer('o2').output)
        # middle_output = model_middle.predict(train)  # 获取该中间层输出
        # print(middle_output.shape)
        # # 保存输出
        # middle_output = pd.DataFrame(middle_output)
        # middle_output.to_csv("./output/middle_output/" + feature_name + '_' + 'dense16_o2' + ".csv")

        model = load_model(model_save_path)
        y_pred = model.predict(test)
    if (model_name == 'bigru'):
        train = train[:, np.newaxis]
        test = test[:, np.newaxis]
        train, val, train_label, val_label = train_test_split(train, train_label, test_size=0.2, random_state=123)
        model_save_path = './model/' + feature_name + "_" + model_name + '.h5'
        model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        model = bigru(train.shape[1])
        #64,128
        model.fit(train, train_label, batch_size=128, epochs=128, shuffle=True, validation_data=(val, val_label)
                  , callbacks=[model_check, reduct_L_rate])
        model = load_model(model_save_path)
        # # # #以中间层建立模型
        # model_middle = Model(inputs=model.input, outputs=model.get_layer('bigru_dense').output)
        # middle_output = model_middle.predict(train)  # 获取该中间层输出
        # # middle_output = middle_output[:, 0, :]  # 保存为2034,16
        # middle_output=middle_output.T
        # middle_output=pd.DataFrame(middle_output)
        # middle_output.to_csv("./output/middle_output/" + feature_name + '_' + 'bigru_dense' + ".csv")
        # print("1")
        # SHAP画图
        import shap
        explainer = shap.GradientExplainer(model, train)
        shap_values = explainer.shap_values(test)
        print("1")

        shap_values_2D = shap_values[0].reshape(-1,20)#461,440
        test_2D=test.reshape(-1,20)

        test_2D = pd.DataFrame(data=test_2D, columns=save_column)
        shap.summary_plot(shap_values_2D, test_2D, plot_size=0.5)


        y_pred = model.predict(test)

    np.savez('./ind_save/' + feature_name + '_' + model_name + '/true_label' + ".npz", test_label)
    np.savez('./ind_save/' + feature_name + '_' + model_name + '/pred_proba' + ".npz", y_pred)
    acc, se, sp, mcc, auROC, precision, f1, AUPRC = eva(y_pred, test_label)
    print(' ACC:', acc)
    print(' SE:', se)
    print(' SP:', sp)
    print(' MCC:', mcc)
    print(' AUROC:', auROC)
    print(' Precision:', precision)
    print(' F1:', f1)
    print(' AUPRC:', AUPRC)

    import csv
    f = open('output/ind_result.csv', 'a')
    writer = csv.writer(f)
    writer.writerow([str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))])
    writer.writerow([feature_name])
    writer.writerow([model_name])
    header = ['sensitivity', 'specificity', "Precision", 'acc',
              'mcc', 'f1', 'AUROC', 'AUPRC']
    writer.writerow(header)
    writer.writerow([str(se), str(sp), str(precision), str(acc),
                     str(mcc), str(f1), str(auROC), str(AUPRC)])
    # close the file
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', help='feature_name')
    parser.add_argument('-m', help='model_name')
    args = parser.parse_args()
    feature_name = args.f
    model_name = args.m

    run(feature_name, model_name)
