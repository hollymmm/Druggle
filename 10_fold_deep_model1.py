import os
import pickle
import random
import pandas as pd
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

def bigru(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(units=64,batch_input_shape=(None,1,input_shape),return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(learning_rate=1e-4)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model
def run(feature_name,model_name,category_feature):
    print(feature_name)
    print(model_name)
    print(category_feature)

    ##################################1.读入特征#################################################3

    path1 = './data/' + feature_name + '/train_' + feature_name + '.csv'
    train = pd.read_csv(path1, header=0, index_col=0)
    save_column = train.columns
    path1 = './data/' + feature_name + '/test_' + feature_name + '.csv'
    test = pd.read_csv(path1, header=0, index_col=0)
    train = np.array(train)
    test = np.array(test)
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

    isExists = os.path.exists('./save/' + feature_name + '_' + model_name)
    if not isExists:  # 判断如果文件不存在,则创建
        os.makedirs('./save/' + feature_name + '_' + model_name)
        print("%s 目录创建成功")
    for epoces in range(10):
        kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=epoces)
        cnt=0
        for train_index,test_index in kf.split(train,train_label):
            print(cnt)
            new_train,new_test=train[train_index],train[test_index]
            train_label2,test_label2=train_label[train_index],train_label[test_index]
            if model_name=='svm':
                model = SVC(probability=True).fit(new_train, train_label2)
                y_pred = model.predict_proba(new_test)[:,1]#255

            if(model_name=='bigru'):
                new_train = new_train[:, np.newaxis]
                new_test = new_test[:, np.newaxis]
                model_save_path = './model/' + feature_name + "_" + model_name + '.h5'
                model_check = ModelCheckpoint(filepath=model_save_path, monitor='val_accuracy', save_best_only=True)
                reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
                model = bigru(new_train.shape[1])
                model.fit(new_train, train_label2, batch_size=64, epochs=128, shuffle=True, validation_data=(new_test,test_label2)
                          , callbacks=[model_check, reduct_L_rate])
                model = load_model(model_save_path)
                y_pred = model.predict(new_test)

            np.savez('./save/'+feature_name+'_'+model_name+'/epoces' + str(epoces) + "_fold" + str(cnt)+'_true_label' + ".npz",test_label2)
            np.savez('./save/'+feature_name+'_'+model_name+'/epoces' + str(epoces) + "_fold" + str(cnt)+'_pred_proba' + ".npz",y_pred)

            y_pred_label=model.predict(new_test)
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
    writer.writerow([feature_name])
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
    parser.add_argument('-c')
    parser.add_argument('-f',help='feature_name')
    parser.add_argument('-m', help='model_name')
    args=parser.parse_args()
    feature_name=args.f
    model_name=args.m
    category_feature = args.c
    run(feature_name,model_name,category_feature)
