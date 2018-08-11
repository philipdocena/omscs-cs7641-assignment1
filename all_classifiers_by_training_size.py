# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:55:28 2018

@author: Philip Docena
"""

# import libraries
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import util


def run_test(test_type='MNIST'):
    """ run test over different training sizes """
    print('Running tests, all classifiers over different training sizes...')
    
    # set default (check scikit for defaults) or simple/reasonable classifier settings
    # replace with tuned settings if desired
    # note that different tuned settings might be needed for best performance on each dataset
    clf1=DecisionTreeClassifier(random_state=0)
    clf2=KNeighborsClassifier(n_neighbors=1)
    clf3=GradientBoostingClassifier(n_estimators=50,learning_rate=1.0,max_depth=10,random_state=0)
    clf4=svm.SVC(kernel='linear',C=1.0,random_state=0)
    clf5=MLPClassifier(hidden_layer_sizes=100,solver='adam',activation='relu',
                       learning_rate='adaptive',random_state=0)
    
    clf_list=[clf1,clf2,clf3,clf4,clf5]
    clf_names=['Decision Tree','kNN','Boosted Trees','SVM','MLP']
    
    if test_type=='MNIST':
        f_names_train=['MNIST_Decision_Trees_by_training_size_vs_train_data.csv',
                       'MNIST_kNN_by_training_size_vs_train_data.csv',
                       'MNIST_Boosted_Trees_by_training_size_vs_train_data.csv',
                       'MNIST_SVM_by_training_size_vs_train_data.csv',
                       'MNIST_MLP_by_training_size_vs_train_data.csv']
        
        f_names_val=['MNIST_Decision_Trees_by_training_size_vs_validation_data.csv',
                     'MNIST_kNN_by_training_size_vs_validation_data.csv',
                     'MNIST_Boosted_Trees_by_training_size_vs_validation_data.csv',
                     'MNIST_SVM_by_training_size_vs_validation_data.csv',
                     'MNIST_MLP_by_training_size_vs_validation_data.csv']
        
        f_names_test=['MNIST_Decision_Trees_by_training_size_vs_test_data.csv',
                      'MNIST_kNN_by_training_size_vs_test_data.csv',
                      'MNIST_Boosted_Trees_by_training_size_vs_test_data.csv',
                      'MNIST_SVM_by_training_size_vs_test_data.csv',
                      'MNIST_MLP_by_training_size_vs_test_data.csv']
    else:
        f_names_train=['USPS_Decision_Trees_by_training_size_vs_train_data.csv',
                       'USPS_kNN_by_training_size_vs_train_data.csv',
                       'USPS_Boosted_Trees_by_training_size_vs_train_data.csv',
                       'USPS_SVM_by_training_size_vs_train_data.csv',
                       'USPS_MLP_by_training_size_vs_train_data.csv']
        
        f_names_val=['USPS_Decision_Trees_by_training_size_vs_validation_data.csv',
                     'USPS_kNN_by_training_size_vs_validation_data.csv',
                     'USPS_Boosted_Trees_by_training_size_vs_validation_data.csv',
                     'USPS_SVM_by_training_size_vs_validation_data.csv',
                     'USPS_MLP_by_training_size_vs_validation_data.csv']
        
        f_names_test=['USPS_Decision_Trees_by_training_size_vs_test_data.csv',
                      'USPS_kNN_by_training_size_vs_test_data.csv',
                      'USPS_Boosted_Trees_by_training_size_vs_test_data.csv',
                      'USPS_SVM_by_training_size_vs_test_data.csv',
                      'USPS_MLP_by_training_size_vs_test_data.csv']
    
    # define train and test size as desired
    train_size=1000
    test_size=1000
    
    for clf,clf_name,f_name_train,f_name_val,f_name_test in zip(clf_list,clf_names,
                                                                f_names_train,
                                                                f_names_val,
                                                                f_names_test):
        print('Running',clf_name,'...')
        err_train_list=[]
        err_val_list=[]
        err_test_list=[]
        
        # test different values for the training size
        train_size_list=np.arange(100,1000,100)
        
        if test_type=='MNIST':
            train,train_label,test,test_label=util.MNIST_loader(1,train_size,1,test_size,echo=False)
        else:
            train,train_label,test,test_label=util.USPS_loader(1,train_size,1,test_size,echo=False)
    
        X_train,X_val,y_train,y_val=train_test_split(train,train_label,test_size=0.2,random_state=0)
        print('... train and val set size',X_train.shape,X_val.shape)
        
        for i in train_size_list:
            clf.fit(X_train[:i],y_train[:i])
            
            acc_train=clf.score(X_train[:i],y_train[:i])
            
            acc_val=clf.score(X_val,y_val)
            acc_test=clf.score(test,test_label)
            
            err_train_list.append(1.0-acc_train)
            err_val_list.append(1.0-acc_val)
            err_test_list.append(1.0-acc_test)
            #print('... train, val, and test accuracy (error rate) at train size:',acc_train,acc_val,acc_test,i)
        
        print('... done, min_train, max_train',np.min(err_train_list),np.max(err_train_list))
        print('... done, min_val, max_val',np.min(err_val_list),np.max(err_val_list))
        print('... done, min_test, max_test',np.min(err_test_list),np.max(err_test_list))
        
        df_train=pd.DataFrame({'Classifier':[clf_name]*len(err_train_list),
                               'Size':train_size_list,
                               'Error':err_train_list})
        df_val=pd.DataFrame({'Classifier':[clf_name]*len(err_val_list),
                               'Size':train_size_list,
                               'Error':err_val_list})
        df_test=pd.DataFrame({'Classifier':[clf_name]*len(err_test_list),
                              'Size':train_size_list,
                              'Error':err_test_list})
        
        df_train.to_csv(f_name_train,index=False,header=True)
        df_val.to_csv(f_name_val,index=False,header=True)
        df_test.to_csv(f_name_test,index=False,header=True)
        
    return f_names_train,f_names_val,f_names_test
    

if __name__=='__main__':
    #util.show_distribution()
    
    fn_train,fn_val,fn_test=run_test('MNIST')
    #fn_train,fn_val,fn_test=run_test('USPS')
    
    util.plotterA(fn_val,fn_train,'training','validation')
    util.plotterA(fn_val,fn_test,'validation','test')
