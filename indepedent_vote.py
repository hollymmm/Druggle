import numpy as np
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


    pre=read('/home/tgliu/PycharmProjects/Druggle2/ind_save/all_20_bigru'+'/pred_proba.npz')#461,1
    pre2=read('/home/tgliu/PycharmProjects/Druggle2/ind_save/combine_all_feature3'+'/pred_proba.npz')
    pre2=pre2[:,np.newaxis]
    pre_all=np.hstack([pre,pre2])


    mean=np.average(pre_all, axis=1,weights=[1,2])

    data2 = np.load('/home/tgliu/PycharmProjects/Druggle2/ind_save/all_20_bigru'+'/true_label.npz')
    tmp = data2.files
    for i in range(len(tmp)):
        true_label = data2[tmp[i]]
    acc, se, sp, mcc, auROC, precision, f1, AUPRC = eva(mean,true_label)
    print(' ACC:', acc)
    print(' SE:', se)
    print(' SP:', sp)
    print(' MCC:', mcc)
    print(' AUROC:', auROC)
    print(' Precision:', precision)
    print(' F1:', f1)
    print(' AUPRC:', AUPRC)

if __name__=='__main__':
    run()
