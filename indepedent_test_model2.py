import os
import pickle
import random
import warnings

import tensorflow
from keras import Sequential, Input, models
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import BatchNormalization, Bidirectional, LSTM, GRU, Convolution1D, Concatenate
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


set_tf_device('gpu') # 'cpu' or 'gpu'

setup_seed(2022);#设置随机种子


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
    sp = recall_score(true_values, y_labels,pos_label=0)
    mcc = matthews_corrcoef(true_values, y_labels)
    precision=precision_score(true_values,y_labels)
    f1=f1_score(true_values,y_labels)
    precision2, recall, thresholds = precision_recall_curve(true_values, y_labels)
    AUPRC = auc(recall, precision2)
    return acc, se, sp, mcc, AUC,precision,f1,AUPRC

def combine_all(input_shape1,input_shape2,
                input_shape4,input_shape5,input_shape6,
                input_shape7):
    a = Input(shape=(1,input_shape1))
    a_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnna')(a)
    a_cnn_bn = BatchNormalization(axis=-1)(a_cnn)
    a_cnn_bn_a = Activation('relu')(a_cnn_bn)
    print(a_cnn_bn_a.shape)
    b = Input(shape=(1,input_shape2))
    b_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnnb')(b)
    b_cnn_bn = BatchNormalization(axis=-1)(b_cnn)
    b_cnn_bn_a = Activation('relu')(b_cnn_bn)
    print(b_cnn_bn_a.shape)
    # c = Input(shape=(1,input_shape3))
    # c_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnnc')(c)
    # c_cnn_bn = BatchNormalization(axis=-1)(c_cnn)
    # c_cnn_bn_a = Activation('relu')(c_cnn_bn)
    # print(c_cnn_bn_a.shape)

    d = Input(shape=(1,input_shape4))
    d_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnnd')(d)
    d_cnn_bn = BatchNormalization(axis=-1)(d_cnn)
    d_cnn_bn_a = Activation('relu')(d_cnn_bn)
    print(d_cnn_bn_a.shape)

    e = Input(shape=(1,input_shape5))
    e_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnne')(e)
    e_cnn_bn = BatchNormalization(axis=-1)(e_cnn)
    e_cnn_bn_a = Activation('relu')(e_cnn_bn)
    print(e_cnn_bn_a.shape)

    f = Input(shape=(1,input_shape6))
    f_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnnf')(f)
    f_cnn_bn = BatchNormalization(axis=-1)(f_cnn)
    f_cnn_bn_a = Activation('relu')(f_cnn_bn)
    print(f_cnn_bn_a.shape)

    g = Input(shape=(1,input_shape7))
    g_cnn = Convolution1D(filters=16, kernel_size=1, padding='same',name='cnng')(g)
    g_cnn_bn = BatchNormalization(axis=-1)(g_cnn)
    g_cnn_bn_a = Activation('relu')(g_cnn_bn)
    print(g_cnn_bn_a.shape)

    mergeInput = Concatenate(axis=-1,name='cnn_concatenate')([a_cnn_bn_a, b_cnn_bn_a,
                                                 d_cnn_bn_a,e_cnn_bn_a,f_cnn_bn_a
                                                 ,g_cnn_bn_a])#16*7=112
    dense1=Dense(1)(mergeInput)
    out=Activation('sigmoid')(dense1)
    model=models.Model(inputs=[a,b,d,e,f,g],outputs=out)
    adam = Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def run(model_name):

    print(model_name)


    ##################################1.读入特征#################################################

    category_feature='feature'
    path1 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_ANOVA_ASDC_220.csv'
    train1 = pd.read_csv(path1, header=0, index_col=0)
    train1= np.array(train1)

    path2 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_ANOVA_DDE_160.csv'
    train2 = pd.read_csv(path2, header=0, index_col=0)
    train2= np.array(train2)

    path3 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_ANOVA_prot_bert_bfd_560.csv'
    train3 = pd.read_csv(path3, header=0, index_col=0)
    train3= np.array(train3)

    path4 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_ANOVA_t5_220.csv'
    train4 = pd.read_csv(path4, header=0, index_col=0)
    train4= np.array(train4)

    path5 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_ANOVA_tape_avg_220.csv'
    train5 = pd.read_csv(path5, header=0, index_col=0)
    train5= np.array(train5)

    path6 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/train_lgbm_esm2_240.csv'
    train6 = pd.read_csv(path6, header=0, index_col=0)
    train6= np.array(train6)

    path7 = '//home/tgliu/PycharmProjects/cnn模型部分/feature/train_lgbm_pssm400_60.csv'
    train7 = pd.read_csv(path7, header=0, index_col=0)
    train7= np.array(train7)

    path1 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_ANOVA_ASDC_220.csv'
    test1 = pd.read_csv(path1, header=0, index_col=0)
    test1 = np.array(test1)

    path2 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_ANOVA_DDE_160.csv'
    test2 = pd.read_csv(path2, header=0, index_col=0)
    test2 = np.array(test2)

    path3 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_ANOVA_prot_bert_bfd_560.csv'
    test3 = pd.read_csv(path3, header=0, index_col=0)
    test3 = np.array(test3)

    path4 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_ANOVA_t5_220.csv'
    test4 = pd.read_csv(path4, header=0, index_col=0)
    test4 = np.array(test4)

    path5 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_ANOVA_tape_avg_220.csv'
    test5 = pd.read_csv(path5, header=0, index_col=0)
    test5 = np.array(test5)

    path6 = '/home/tgliu/PycharmProjects/cnn模型部分/feature/test_lgbm_esm2_240.csv'
    test6 = pd.read_csv(path6, header=0, index_col=0)
    test6 = np.array(test6)

    path7 = '//home/tgliu/PycharmProjects/cnn模型部分/feature/test_lgbm_pssm400_60.csv'
    test7 = pd.read_csv(path7, header=0, index_col=0)
    test7 = np.array(test7)


    pos_num = 1224
    neg_num = 1319
    train_label = np.array([1] * pos_num + [0] * neg_num)
    pos_num = 224
    neg_num = 237
    test_label = np.array([1] * pos_num + [0] * neg_num)



    ###############################2.训练####################################################
    isExists = os.path.exists('./ind_save/' +   model_name)
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs('./ind_save/' +  model_name)
        print("%s 目录创建成功")
    if(model_name!='combine_all'):
        train1,val1,train_label1,val_label1=train_test_split(train1,train_label,test_size=0.2, random_state=123)
        train2, val2, train_label2, val_label2 = train_test_split(train2, train_label, test_size=0.2, random_state=123)
        train3, val3, train_label3, val_label3 = train_test_split(train3, train_label, test_size=0.2, random_state=123)
        train4, val4, train_label4, val_label4 = train_test_split(train4, train_label, test_size=0.2, random_state=123)
        train5, val5, train_label5, val_label5 = train_test_split(train5, train_label, test_size=0.2, random_state=123)
        train6, val6, train_label6, val_label6 = train_test_split(train6, train_label, test_size=0.2, random_state=123)
        train7, val7, train_label7, val_label7 = train_test_split(train7, train_label, test_size=0.2, random_state=123)

        model_save_path = './model/' + model_name + '.h5'
        model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
        reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
        #num,input_shape1,input_shape2,input_shape3

        model = combine_all(train1.shape[1],train2.shape[1],train4.shape[1],
                            train5.shape[1],train6.shape[1],train7.shape[1])
        train1=train1[:,np.newaxis,:]
        val1=val1[:,np.newaxis,:]
        test1=test1[:,np.newaxis,:]
        train2 = train2[:, np.newaxis, :]
        val2 = val2[:, np.newaxis, :]
        test2 = test2[:, np.newaxis, :]
        train3 = train3[:, np.newaxis, :]
        val3 = val3[:, np.newaxis, :]
        test3 = test3[:, np.newaxis, :]
        train4 = train4[:, np.newaxis, :]
        val4 = val4[:, np.newaxis, :]
        test4 = test4[:, np.newaxis, :]
        train5 = train5[:, np.newaxis, :]
        val5 = val5[:, np.newaxis, :]
        test5 = test5[:, np.newaxis, :]
        train6 = train6[:, np.newaxis, :]
        val6 = val6[:, np.newaxis, :]
        test6 = test6[:, np.newaxis, :]
        train7 = train7[:, np.newaxis, :]
        val7 = val7[:, np.newaxis, :]
        test7 = test7[:, np.newaxis, :]

        model.fit([train1,train2,train4,train5,train6,train7],train_label1, batch_size=64, epochs=128, shuffle=True,
                  validation_data=([val1,val2,val4,val5,val6,val7], val_label1)
                  , callbacks=[model_check, reduct_L_rate])
        model = load_model(model_save_path)

        # #以中间层建立模型
        # model_middle = Model(inputs=model.input, outputs=model.get_layer('cnnb').output)
        # middle_output=model_middle.predict([train_all_29,train_seqvec_all_esm,train_t5_esm])#获取该中间层输出
        # middle_output=middle_output[:,0,:]   #保存为2034,48
        # #保存输出
        # middle_output=pd.DataFrame(middle_output)
        # middle_output.to_csv("./output/middle_output/"+feature_name+'_'+'_cnnb'+".csv")
        #
        # #以中间层建立模型
        model_middle = Model(inputs=model.input, outputs=model.get_layer('cnn_concatenate').output)
        middle_output=model_middle.predict([train1,train2,train4,train5,train6,train7])#获取该中间层输出
        middle_output=middle_output[:,0,:]   #保存为2034,48
        middle_output=middle_output.T
        middle_output=pd.DataFrame(middle_output)
        #保存输出
        middle_output=pd.DataFrame(middle_output)
        middle_output.to_csv("./output/middle_output/"+'cnn_concatenate'+".csv")


        #461,1,1
        y_pred = model.predict([test1,test2,test4,test5,test6,test7])
        #461
        y_pred=np.reshape(y_pred,-1)


    np.savez('./ind_save/' + model_name + '/true_label' + ".npz", test_label)
    np.savez('./ind_save/'+ model_name + '/pred_proba' + ".npz", y_pred)
    acc, se, sp, mcc, auROC, precision, f1, AUPRC = eva(y_pred,test_label)
    print(' ACC:',acc)
    print(' SE:',se)
    print(' SP:',sp)
    print(' MCC:',mcc)
    print(' AUROC:',auROC)
    print(' Precision:',precision)
    print(' F1:',f1)
    print(' AUPRC:',AUPRC)

    import csv
    f = open('output/ind_result.csv', 'a')
    writer = csv.writer(f)
    writer.writerow([str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))])
    writer.writerow([model_name])
    header=['sensitivity','specificity',"Precision",'acc',
                'mcc','f1','AUROC','AUPRC']
    writer.writerow(header)
    writer.writerow([str(se),str(sp),str(precision),str(acc),
                str(mcc),str(f1),str(auROC),str(AUPRC)])
    # close the file
    f.close()
    



if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()


    parser.add_argument('-m', help='model_name')
    args=parser.parse_args()

    model_name=args.m
    run(model_name)
