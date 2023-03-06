import os
def get_indepentdet_res():
    run='python indepedent_test_model1.py'+' -f all_20 -m bigru'
    os.system(run)
    run = 'python indepedent_test_model2.py' + ' -m combine_all_feature3'
    os.system(run)
def get_cv_res():
    run = 'python 10_fold_deep_model2.py' + ' -m combine_all_feature3'
    os.system(run)
    run = 'python 10_fold_deep_model1.py' + ' -f all_20 -m bigru'
    os.system(run)
def indepedent_vote():
    run = 'python indepedent_vote.py'
    os.system(run)
def cv_vote():
    run='python 10_fold_vote.py'
    os.system(run)
if __name__=='__main__':
    indepedent_vote()
