# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:13:19 2018

@author: Philip Docena
"""

# import libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import util


def run_test():
    """ set up a series of CV grid searches """
    train_size=1000
    test_size=1000
    
    #test_types=['USPS']
    #test_types=['MNIST']
    test_types=['MNIST','USPS']
    
    clf_type='Boosted_Trees'
    rerun_CV=True
    #rerun_CV=False
    
    # setup grid search parameters
    # intentionally incomplete and restricted, change as desired
    num_cv_folds=10
    param_names=['base_estimator__max_depth','base_estimator__min_samples_leaf']
    param_values=[range(1,6,1),range(1,6,1)]
    
    print('Running',clf_type,'CV grid search tests...')
    for test_type in test_types:
        print('Running CV on dataset',test_type,'...')
        if test_type=='MNIST':
            train,train_label,_,_=util.MNIST_loader(1,train_size,1,test_size,echo=False)
        else:
            train,train_label,_,_=util.USPS_loader(1,train_size,1,test_size,echo=False)
        
        for param_name,param_value in zip(param_names,param_values):
            print('... on parameter',param_name)
            if rerun_CV:
                params={param_name:param_value}
                # check unlisted default settings vs intended analysis
                clf_cv=GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=10,
                                                       learning_rate=1.0,random_state=0),
                                    param_grid=params,cv=num_cv_folds,verbose=1)
                util.run_CV(clf_cv,clf_type,test_type,train,train_label,param_name,param_value)
            
            # plot from files
            util.plotterB(str(clf_type+'_grid_search_'+param_name+'_mean_'+test_type+'.csv'),
                          str(clf_type+'_grid_search_'+param_name+'_mean_std_'+test_type+'.csv'),
                          str(param_name+' ('+test_type+')'),str('Accuracy ('+test_type+')'))

if __name__=='__main__':
    run_test()
