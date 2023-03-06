#要改成-feature3的
import os
import pickle
import random
import pandas as pd
from keras import models
from scipy.stats import sem
import tensorflow
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from numpy.random import seed
from sklearn.svm import SVC
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers  import *
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from numpy.random import seed
from tensorflow.keras import backend as K
import time




def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tensorflow.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

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

    mergeInput = Concatenate(axis=-1,name='o1')([a_cnn_bn_a, b_cnn_bn_a,
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


    ##################################1.读入特征#################################################3


    category_feature='feature'
    path1 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_ANOVA_ASDC_220.csv'
    train1 = pd.read_csv(path1, header=0, index_col=0)
    train1= np.array(train1)

    path2 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_ANOVA_DDE_160.csv'
    train2 = pd.read_csv(path2, header=0, index_col=0)
    train2= np.array(train2)

    path3 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_ANOVA_prot_bert_bfd_560.csv'
    train3 = pd.read_csv(path3, header=0, index_col=0)
    train3= np.array(train3)

    path4 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_ANOVA_t5_220.csv'
    train4 = pd.read_csv(path4, header=0, index_col=0)
    train4= np.array(train4)

    path5 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_ANOVA_tape_avg_220.csv'
    train5 = pd.read_csv(path5, header=0, index_col=0)
    train5= np.array(train5)

    path6 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_lgbm_esm2_240.csv'
    train6 = pd.read_csv(path6, header=0, index_col=0)
    train6= np.array(train6)

    path7 = '/home/tgliu/PycharmProjects/Druggle2/feature/train_lgbm_pssm400_60.csv'
    train7 = pd.read_csv(path7, header=0, index_col=0)
    train7= np.array(train7)

    path1 = '/home/tgliu/PycharmProjects/Druggle2/feature/test_ANOVA_ASDC_220.csv'
    test1 = pd.read_csv(path1, header=0, index_col=0)
    test1 = np.array(test1)

    path2 = '/home/tgliu/PycharmProjects/Druggle2/feature/test_ANOVA_DDE_160.csv'
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

    ###############################2.训练####################################################
    # #
    from sklearn.model_selection import StratifiedKFold
    all_acc = []
    all_se = []
    all_sp = []
    all_mcc = []
    all_auROC = []
    all_pre = []
    all_f1 = []
    all_auprc = []

    isExists = os.path.exists('/home/tgliu/PycharmProjects/Druggle2/save' +model_name)
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs('/home/tgliu/PycharmProjects/Druggle2/save' +  model_name)
        print("%s 目录创建成功")
    for epoces in range(10):
        kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=epoces)
        cnt=0
        for train_index,test_index in kf.split(train1,train_label):
            print(cnt)
            new_train1,new_test1=train1[train_index],train1[test_index]
            new_train2,new_test2 = train2[train_index], train2[test_index]
            new_train3 ,new_test3= train3[train_index], train3[test_index]
            new_train4 ,new_test4= train4[train_index], train4[test_index]
            new_train5 ,new_test5= train5[train_index], train5[test_index]
            new_train6 ,new_test6= train6[train_index], train6[test_index]
            new_train7,new_test7 = train7[train_index], train7[test_index]
            train_label2,test_label2=train_label[train_index],train_label[test_index]

            if(model_name=='combine_all_feature3'):


                new_train1=new_train1[:,np.newaxis]
                new_train2 = new_train2[:, np.newaxis]
                new_train3 = new_train3[:, np.newaxis]
                new_train4 = new_train4[:, np.newaxis]
                new_train5 = new_train5[:, np.newaxis]
                new_train6 = new_train6[:, np.newaxis]
                new_train7 = new_train7[:, np.newaxis]
                new_test1 = new_test1[:, np.newaxis]
                new_test2 = new_test2[:, np.newaxis]
                new_test3 = new_test3[:, np.newaxis]
                new_test4 = new_test4[:, np.newaxis]
                new_test5 = new_test5[:, np.newaxis]
                new_test6 = new_test6[:, np.newaxis]
                new_test7 = new_test7[:, np.newaxis]

                model_save_path = './model/' +   model_name + '.h5'
                model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
                reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
                model = combine_all(new_train1.shape[2],new_train2.shape[2],
                                    new_train4.shape[2],new_train5.shape[2],new_train6.shape[2],
                                    new_train7.shape[2])
                model.fit([new_train1,new_train2,
                                    new_train4,new_train5,new_train6,
                                    new_train7], train_label2, batch_size=64, epochs=128, shuffle=True,
                          validation_data=([new_test1,new_test2,
                        new_test4,new_test5,new_test6,new_test7], test_label2)
                          , callbacks=[model_check, reduct_L_rate])
                model = load_model(model_save_path)
                y_pred = model.predict([new_test1,new_test2,new_test4,new_test5,new_test6,new_test7])
                y_pred = np.reshape(y_pred, -1)

            np.savez('/home/tgliu/PycharmProjects/Druggle2/save'+model_name+'/epoces' + str(epoces) + "_fold" + str(cnt)+'_true_label' + ".npz",test_label2)
            np.savez('/home/tgliu/PycharmProjects/Druggle2/save'+model_name+'/epoces' + str(epoces) + "_fold" + str(cnt)+'_pred_proba' + ".npz",y_pred)

            acc, se, sp, mcc, auROC, precision, f1, AUPRC = eva(y_pred,test_label2)
            cnt = cnt + 1


            all_acc.append(acc)
            all_se.append(se)
            all_sp.append(sp)
            all_mcc.append(mcc)
            all_auROC.append(auROC)
            all_pre.append(precision)
            all_f1.append(precision)
            all_auprc.append(AUPRC)



    #5折结束后，计算mean
    meanACC=np.mean(all_acc)
    meanSE=np.mean(all_se)
    meanSP=np.mean(all_sp)
    meanMCC=np.mean(all_mcc)
    meanAUROC=np.mean(all_auROC)
    meanPrecision=np.mean(all_pre)
    meanF1=np.mean(all_f1)
    meanAUPRC=np.mean(all_auprc)

    stdACC = np.std(all_acc)
    stdSE = np.std(all_se)
    stdSP = np.std(all_sp)
    stdMCC = np.std(all_mcc)
    stdAUROC = np.std(all_auROC)
    stdPrecision = np.std(all_pre)
    stdF1 = np.std(all_f1)
    stdAUPRC = np.std(all_auprc)



    print('meanACC:',meanACC)
    print('meanSE:',meanSE)
    print('meanSP:',meanSP)
    print('meanMCC:',meanMCC)
    print('meanAUROC:',meanAUROC)
    print('meanPrecision:',meanPrecision)
    print('meanF1:',meanF1)
    print('meanAUPRC:',meanAUPRC)
    #std
    print('stdACC:', stdACC)
    print('stdSE:', stdSE)
    print('stdSP:', stdSP)
    print('stdMCC:', stdMCC)
    print('stdAUROC:', stdAUROC)
    print('stdPrecision:', stdPrecision)
    print('stdF1:', stdF1)
    print('stdAUPRC:', stdAUPRC)



    import csv
    f = open('output/10foldsdeep_result.csv', 'a')
    writer = csv.writer(f)
    writer.writerow([str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))])
    writer.writerow(["all_three"])
    writer.writerow([model_name])
    header=['sensitivity','specificity',"Precision",'acc',
                'mcc','f1','AUROC','AUPRC']
    writer.writerow(header)
    writer.writerow([str(meanSE),str(meanSP),str(meanPrecision),str(meanACC),
                str(meanMCC),str(meanF1),str(meanAUROC),str(meanAUPRC)])
    writer.writerow([str(stdSE) , str(stdSP) ,str(stdPrecision) ,str(stdACC) ,
                str(stdMCC) , str(stdF1) , str(stdAUROC) ,str(stdAUPRC)])
    # close the file
    f.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()


    parser.add_argument('-m', help='model_name')
    args=parser.parse_args()

    model_name=args.m

    run(model_name)

