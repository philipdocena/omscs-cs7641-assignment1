# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 00:25:01 2018

@author: Philip Docena
"""

# import libraries
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import util


def run_test():
    """ set up a series of CV grid searches """
    train_size=1000
    test_size=1000
    
    #test_types=['USPS']
    #test_types=['MNIST']
    test_types=['MNIST','USPS']
    
    clf_type='kNN'
    rerun_CV=True
    #rerun_CV=False
    
    # setup grid search parameters
    # intentionally incomplete and restricted, change as desired
    num_cv_folds=10
    param_names=['n_neighbors','weights']
    param_values=[range(1,6,1),['uniform','distance']]
    param_string_types=[False,True]
    
    print('Running',clf_type,'CV grid search tests...')
    for test_type in test_types:
        print('Running CV on dataset',test_type,'...')
        if test_type=='MNIST':
            train,train_label,_,_=util.MNIST_loader(1,train_size,1,test_size,echo=False)
        else:
            train,train_label,_,_=util.USPS_loader(1,train_size,1,test_size,echo=False)
        
        for param_name,param_value,param_str_type in zip(param_names,param_values,param_string_types):
            print('... on parameter',param_name)
            if rerun_CV:
                params={param_name:param_value}
                np.random.seed(0)   # need this, no random_state on CV and kNN
                # check unlisted default settings vs intended analysis
                # default n_neighbors=3 for the weights cv
                clf_cv=GridSearchCV(KNeighborsClassifier(algorithm='ball_tree',n_neighbors=3),
                                    param_grid=params,cv=num_cv_folds,verbose=1)
                util.run_CV(clf_cv,clf_type,test_type,train,train_label,param_name,param_value)
            
            # plot from files
            util.plotterB(str(clf_type+'_grid_search_'+param_name+'_mean_'+test_type+'.csv'),
                          str(clf_type+'_grid_search_'+param_name+'_mean_std_'+test_type+'.csv'),
                          str(param_name+' ('+test_type+')'),str('Accuracy ('+test_type+')'),
                          string=param_str_type)


if __name__=='__main__':
    run_test()
