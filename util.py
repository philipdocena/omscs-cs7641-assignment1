# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 14:07:27 2018
‏
@author: Philip Docena
"""

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digits_loader(train_filename,train_start,train_end,test_filename,test_start,test_end,echo=False):
    """ load digits dataset """
    if echo: print('Loading digits file...')
    
    train=pd.read_csv(train_filename,dtype='uint8').iloc[train_start-1:train_end].values
    test=pd.read_csv(test_filename,dtype='uint8').iloc[test_start-1:test_end].values
    
    train,train_label=train[:,1:],train[:,0]
    test,test_label=test[:,1:],test[:,0]
    
    if echo:
        print('... train dimensions:',train.shape)
        print('... test dimensions:',test.shape)
    
    return train,train_label,test,test_label


def MNIST_loader(train_start,train_end,test_start,test_end,echo=False):
    """ prep MNIST dataset filenames """
    if echo: print('Loading MNIST, [train_start, train_end],[test_start, test_end]',train_start,train_end,test_start,test_end)
    
    train_file='data/mnist_train.csv'
    test_file='data/mnist_test.csv'
    
    return digits_loader(train_file,train_start,train_end,test_file,test_start,test_end,echo=echo)


def USPS_loader(train_start,train_end,test_start,test_end,echo=False):
    """ prep USPS dataset filenames """
    if echo: print('Loading USPS, [train_start, train_end],[test_start, test_end]',train_start,train_end,test_start,test_end)
    
    train_file='data/usps_train.csv'
    test_file='data/usps_test.csv'
    
    return digits_loader(train_file,train_start,train_end,test_file,test_start,test_end,echo=echo)


def plotterA(f_names1,f_names2,txt1,txt2):
    """ generate accuracy plots """
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    err_list1=[]
    for fn in f_names1:
        if fn is None:
            print('ERROR: File',fn,'not found.  Make sure test files are available.  Ending run.')
            exit(0)
        
        df=pd.read_csv(fn)
        err_list1.append(df['Error'].values)
        x_range=df['Size'].values
    
    err_list2=[]
    for fn in f_names2:
        if fn is None:
            print('ERROR: File',fn,'not found.  Make sure test files are available.  Ending run.')
            exit(0)
        
        df=pd.read_csv(fn)
        err_list2.append(df['Error'].values)
        x_range=df['Size'].values    
    
    #txt='training'
    ax.plot(x_range,err_list1[0],'k-',marker='+',color='red',label=str('Decision Tree '+txt1+' error'))
    ax.plot(x_range,err_list2[0],'k-.',marker='+',color='red',label=str('Decision Tree '+txt2+' error'))
    
    ax.plot(x_range,err_list1[1],'k-',marker='x',color='green',label=str('kNN '+txt1+' error'))
    ax.plot(x_range,err_list2[1],'k-.',marker='x',color='green',label=str('kNN '+txt2+' error'))
    
    ax.plot(x_range,err_list1[2],'k-',marker='^',color='blue',label=str('Boosted Trees '+txt1+' error'))
    ax.plot(x_range,err_list2[2],'k-.',marker='^',color='blue',label=str('Boosted Trees '+txt2+' error'))
    
    ax.plot(x_range,err_list1[3],'k-',marker='*',color='orange',label=str('SVM '+txt1+' error'))
    ax.plot(x_range,err_list2[3],'k-.',marker='*',color='orange',label=str('SVM '+txt2+' error'))
    
    ax.plot(x_range,err_list1[4],'k-',marker='o',color='magenta',label=str('MLP '+txt1+' error'))
    ax.plot(x_range,err_list2[4],'k-.',marker='o',color='magenta',label=str('MLP '+txt2+' error'))
    
    ax.set_ylim((-0.1, 1.0))
    ax.set_xlabel('Training size')
    ax.set_ylabel('Error rate')
    ax.grid()

    leg=ax.legend(loc='best',fontsize='x-small',fancybox=True)
    leg.get_frame().set_alpha(0.7)

    plt.show()


def plotterB(scores_fname,scores_std_fname,xlabel,ylabel,ylim=None,string=False,log_scale=False):
    """ plot CV results """
    if scores_fname is None or scores_std_fname is None:
        print('ERROR: At least one of the CSV files (',scores_fname,',',scores_std_fname,') was not found.')
        print('Make sure test files are available.  Ending run.')
        exit(0)
    
    df=pd.read_csv(scores_fname)
    scores=df['CV Score'].values
    params=df['Parameter'].values
    
    df=pd.read_csv(scores_std_fname)
    scores_std=df['CV Score StDev'].values
    
    ave_std=np.mean(scores_std)
    
    if string:
        params_old=params
        params=np.arange(len(params))
        plt.bar(range(len(scores)),scores)
        #plt.xticks(np.arange(len(params)),params_old,rotation=45)
        plt.xticks(np.arange(len(params)),params_old)
    else:
        plt.plot(params,scores,label='CV accuracy',marker='x',color='orange')
        plt.fill_between(params,scores-scores_std,scores+scores_std,alpha=0.2,color='b')
        plt.fill_between(params,scores-ave_std,scores+ave_std,alpha=0.4,color='b')
        plt.plot(params,scores-ave_std,'r--',alpha=0.8,label='Mean CV CI')
        plt.plot(params,scores+ave_std,'r--',alpha=0.8)
        
        best_index=np.argmax(scores)
        best_score=scores[best_index]
        
        plt.plot([params[best_index],]*2,[0,best_score],linestyle='-.',marker='x',markeredgewidth=3,ms=8,label='Best CV')
        plt.annotate('(%s, %0.3f)'%(params[best_index],best_score),(params[best_index],best_score))
        
        if log_scale: plt.xscale('log')
        if ylim is not None: plt.ylim(ylim)   
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)    
    plt.legend(loc="best")
    plt.grid()
    
    plt.show()


def show_distribution():
    """ calculate distribution of digits """
    
    train_mnist,train_mnist_label,test_mnist,test_mnist_label=MNIST_loader(1,60000,1,10000,True)
    #train_usps,train_usps_label,test_usps,test_usps_label=USPS_loader(1,7291,1,2007,True)
    
    # show digits distribution
    count=[]
    for i in range(10):
        count.append(len(train_mnist[train_mnist_label==i]))
    print('Count per digit (MNIST):',count)
    print('Percentage per digit (MNIST):',np.array(count)/sum(np.array((count))))
    
    # show digits distribution
    count=[]
    for i in range(10):
        count.append(len(test_mnist[test_mnist_label==i]))
    print('Count per digit (USPS):',count)
    print('Percentage per digit (USPS):',np.array(count)/sum(np.array((count))))


def run_CV(clf,clf_type,test_type,X,y,param_name,param_value):
    """ run CV grid search over a defined set of parameter values and record output to files """
    train_size=len(X)
    
    #params={param_name:param_range}
    #clf=GridSearchCV(DecisionTreeClassifier(random_state=0,criterion='gini'),param_grid=params,cv=num_cv_folds)
    
    clf.fit(X,y)

    scores=clf.cv_results_['mean_test_score']
    scores_std=clf.cv_results_['std_test_score']

    # convert to dataframes
    df_cv_scores=pd.DataFrame({'Classifier':[clf_type]*len(scores),
                               'Size':[train_size]*len(scores),
                               'Parameter':param_value,
                               'CV Score':scores})
    df_cv_scores_std=pd.DataFrame({'Classifier':[clf_type]*len(scores),
                                   'Size':[train_size]*len(scores),
                                   'Parameter':param_value,
                                   'CV Score StDev':scores_std})

    # save to files
    df_cv_scores.to_csv(str(clf_type+'_grid_search_'+param_name+'_mean_'+test_type+'.csv'),index=False,header=True)
    df_cv_scores_std.to_csv(str(clf_type+'_grid_search_'+param_name+'_mean_std_'+test_type+'.csv'),index=False,header=True)


if __name__=='__main__':
    print('helper functions for tests')