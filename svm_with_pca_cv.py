# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:58:38 2018

@author: Philip Docena
"""

# import libraries
import numpy as np

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

import util


def run_test():
    """ set up a series of CV grid searches """
    # reduced size to shorten execution time, change as desired
    train_size=500
    test_size=500
    
    #test_types=['USPS']
    #test_types=['MNIST']
    test_types=['MNIST','USPS']
    
    clf_type='SVM'
    rerun_CV=True
    #rerun_CV=False
    
    # setup grid search parameters
    # intentionally incomplete and restricted, change as desired
    # PCA is to shorten run time, but might not be advisable for analysis
    # reduced cv to shorten execution time, change as desired
    num_cv_folds=5
    num_pca=None
    #num_pca=30
    param_names=['estimator__kernel','estimator__C']
    param_values=[['linear','rbf','poly'],np.logspace(0,5,10)]
    param_string_types=[True,False]
    
    print('Running',clf_type,'CV grid search tests...')
    print('... some settings might take a very long time!')
    for test_type in test_types:
        print('Running CV on dataset',test_type,'...')
        if test_type=='MNIST':
            train,train_label,_,_=util.MNIST_loader(1,train_size,1,test_size,echo=False)
        else:
            train,train_label,_,_=util.USPS_loader(1,train_size,1,test_size,echo=False)
        
        # some datasets/settings might need PCA to shorten execution
        print('... running PCA pre-processing')
        if num_pca is not None:
            pca=PCA(n_components=num_pca)
            train=pca.fit_transform(train)
        
        for param_name,param_value,param_str_type in zip(param_names,param_values,param_string_types):
            print('... on parameter',param_name)
            if rerun_CV:
                params={param_name:param_value}
                # check unlisted default settings vs intended analysis
                clf_cv=GridSearchCV(OneVsRestClassifier(estimator=SVC(C=1.0,degree=3,random_state=0)),
                                    param_grid=params,cv=num_cv_folds,verbose=2)
                util.run_CV(clf_cv,clf_type,test_type,train,train_label,param_name,param_value)
            
            # plot from files
            util.plotterB(str(clf_type+'_grid_search_'+param_name+'_mean_'+test_type+'.csv'),
                          str(clf_type+'_grid_search_'+param_name+'_mean_std_'+test_type+'.csv'),
                          str(param_name+' ('+test_type+')'),str('Accuracy ('+test_type+')'),
                          string=param_str_type,log_scale=True)


if __name__=='__main__':
    run_test()
