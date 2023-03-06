import numpy as np
from scipy.stats import sem
from sklearn.metrics import *

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
def read(fath_name):
    data=np.load(fath_name)
    tmp = data.files
    for i in range(len(tmp)):
        pred = data[tmp[i]]
    return pred
def run():
    acc_score = []
    se_score = []
    sp_score = []
    mcc_score = []
    auROC_score = []
    precision_scores = []
    f1_scores = []
    AUPRC_score = []
    for epoces in range(10):
        print("现在的轮数: ",epoces)
        #每一次十折
        for i in range(10):
            print("第",i,end="")
            print("折")
        #每次10折的每一折
        #把每一折的预测结果连接起来
            #读取
            all_20_pred=np.load('/home/tgliu/PycharmProjects/Druggle2/save/all_20_bigru/epoces'+str(epoces)+'_fold'+str(i)+'_pred_proba.npz')['arr_0']
            combine_all_feature3_pred=np.load('/home/tgliu/PycharmProjects/Druggle2/save/combine_all_feature3/epoces'+str(epoces)+'_fold'+str(i)+'_pred_proba.npz')['arr_0']
            combine_all_feature3_pred=combine_all_feature3_pred[:,np.newaxis]
            #拼接
            fold=np.hstack([all_20_pred,combine_all_feature3_pred])
            #取平均
            fold=np.mean(fold,axis=1)#255
            #载入真值
            true_label=np.load('/home/tgliu/PycharmProjects/Druggle2/save/all_20_bigru/epoces'+str(epoces)+'_fold'+str(i)+'_true_label.npz')['arr_0']
            acc, se, sp, mcc, auROC, precision, f1, AUPRC = eva(fold, true_label)
            acc_score.append(acc)
            se_score.append(se)
            sp_score.append(sp)
            mcc_score.append(mcc)
            auROC_score.append(auROC)
            precision_scores.append(precision)
            f1_scores.append(f1)
            AUPRC_score.append(AUPRC)
    all_acc = np.mean(acc_score)
    print("acc:",all_acc)
    std_acc = sem(acc_score)
    print("acc_std",std_acc)
    all_sp = np.mean(sp_score)
    print("sp",all_sp)
    std_sp = sem(sp_score)
    print("sp_std",std_sp)
    all_sn = np.mean(se_score)
    print('sn',all_sn)
    std_sn = sem(se_score)
    print("std_sn",std_sn)
    all_mcc = np.mean(mcc_score)
    print("mcc",all_mcc)
    std_mcc = sem(mcc_score)
    print('std_mcc',std_mcc)
    all_precision = np.mean(precision_scores)
    print('precision',all_precision)
    std_precision = sem(precision_scores)
    print('std_precision',std_precision)
    all_f1 = np.mean(f1_scores)
    print('f1',all_f1)
    std_f1 = sem(f1_scores)
    print('std_f1',std_f1)
    all_auprc = np.mean(AUPRC_score)
    print('auprc',all_auprc)
    std_auprc = sem(AUPRC_score)
    print('std_auprc',std_auprc)
    all_auc = np.mean(auROC_score)
    print('all_auc',all_auc)
    std_auc = sem(auROC_score)
    print('std_auc',std_auc)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    args=parser.parse_args()

    run()
