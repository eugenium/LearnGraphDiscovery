from sklearn.covariance.empirical_covariance_ import log_likelihood,fast_logdet
from sklearn.covariance import LedoitWolf,GraphLassoCV,EmpiricalCovariance,graph_lasso
from sklearn.covariance.graph_lasso_ import graph_lasso_path
import numpy as np

from scipy import stats
evalonly=True #R dependencies
if(not evalonly):
    import rpy2
    import readline
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    bdgraph=importr("BDgraph")
    rags2ridges=importr("rags2ridges")
    
    from GenSynthCov_Laplace import Gen_Laplace,generate_signal_from_covariance

def toPartialCorr(Prec):
    D=Prec[np.diag_indices(Prec.shape[0])]
    P=Prec.copy()
    D=np.outer(D,D)
    D[D == 0] = 1
    if(np.sum(D==0)):
        print 'Ds',np.sum(D==0)
    if(np.sum(D<0)):
        print 'Dsminus',np.sum(D<0)        
    return -P/np.sqrt(D)

def applyNNBig(Xin,model,msize=500,start=150):
    #Returns an adjacency matrix
    n_features=Xin.shape[1]
    X=Xin.copy()
    #Center
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    larger=np.zeros((msize,msize))
    larger[start:start+n_features,start:start+n_features]=X.T.dot(X)/X.shape[0]
    emp_cov_matrix=np.expand_dims(larger,0)
    
    pred=model.predict(np.expand_dims(emp_cov_matrix,0))
    pred=pred.reshape(msize,msize)[start:start+n_features,start:start+n_features]
    C=np.zeros((X.shape[1],X.shape[1]))
    C[np.triu_indices(n_features,k=1)]=pred[np.triu_indices(n_features,k=1)]
    C=C+C.T
    return C
def applyNNBigRandom(Xin,model,reps=10,msize=500,start=150):
    #Returns an adjacency matrix
    n_features=Xin.shape[1]
    
    X=Xin.copy()
    #Center
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    C_Final=np.zeros((X.shape[1],X.shape[1]))
    ind=np.arange(0,X.shape[1])
    larger=np.zeros((msize,msize))
    for i in xrange(reps):      
        np.random.shuffle(ind)
        I=np.eye(X.shape[1])
        P=I[:,ind]
        larger[start:start+n_features,start:start+n_features]=P.T.dot(X.T.dot(X)).dot(P)/X.shape[0]
        emp_cov_matrix=np.expand_dims(larger,0)
        pred=model.predict(np.expand_dims(emp_cov_matrix,0))
        pred=pred.reshape(msize,msize)[start:start+n_features,start:start+n_features]
        C=np.zeros((X.shape[1],X.shape[1]))
        C[np.triu_indices(n_features,k=1)]=pred[np.triu_indices(n_features,k=1)]
        C=C+C.T
        C=P.dot(C).dot(P.T)
        C_Final+=C
    C_Final=C_Final/float(reps)
    return C_Final
    
def applyNN(Xin,model):
    #Returns an adjacency matrix
    n_features=Xin.shape[1]
    X=Xin.copy()
    #Center
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    emp_cov_matrix=np.expand_dims(X.T.dot(X)/X.shape[0],0)
    
    
    pred=model.predict(np.expand_dims(emp_cov_matrix,0))
    pred=pred.reshape(n_features,n_features)
    C=np.zeros((X.shape[1],X.shape[1]))
    C[np.triu_indices(n_features,k=1)]=pred[np.triu_indices(n_features,k=1)]
    C=C+C.T
    return C
def applyNNRandom(Xin,model,reps=10):
    #Returns an adjacency matrix
    n_features=Xin.shape[1]
    
    X=Xin.copy()
    #Center
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    C_Final=np.zeros((X.shape[1],X.shape[1]))
    ind=np.arange(0,X.shape[1])
    for i in xrange(reps):      
        np.random.shuffle(ind)
        I=np.eye(X.shape[1])
        P=I[:,ind]
        emp_cov_matrix=np.expand_dims(P.T.dot(X.T.dot(X)).dot(P)/X.shape[0],0)
        pred=model.predict(np.expand_dims(emp_cov_matrix,0))
        pred=pred.reshape(n_features,n_features)
        C=np.zeros((X.shape[1],X.shape[1]))
        C[np.triu_indices(n_features,k=1)]=pred[np.triu_indices(n_features,k=1)]
        C=C+C.T
        C=P.dot(C).dot(P.T)
        C_Final+=C
    C_Final=C_Final/float(reps)
    return C_Final

def glassoCVPartial(X):
    gl2=GraphLassoCV().fit(X)
    return np.abs(toPartialCorr(gl2.precision_))

def glassoBonaFidePartial(gl,X,TrueCov):
    #take a 
    ep=EmpiricalCovariance().fit(X)
    emp_cov=ep.covariance_
    _,precs=graph_lasso_path(X, gl.cv_alphas_)
    best_score = -np.inf
    best_ind=0
    for i in xrange(len(gl.cv_alphas_)):
        try:
            this_score = log_likelihood(TrueCov, precs[i])
            if this_score >= .1 / np.finfo(np.float64).eps:
                this_score = np.nan
            if(this_score>best_score):
                best_score=this_score
                best_ind=i
        except:
            print 'exited:',best_score
            continue
    covariance_, precision_, n_iter_ = graph_lasso(
            emp_cov, alpha=gl.cv_alphas_[best_ind], mode=gl.mode, tol=gl.tol*5., max_iter=gl.max_iter, return_n_iter=True)
    return np.abs(toPartialCorr(precision_))

def ldWolfPartial(X):
    L=LedoitWolf().fit(X)
    return np.abs(toPartialCorr(L.precision_))
def RandAdjc(X):
    n_features=X.shape[1]
    R=np.random.rand(n_features,n_features)
    R=R+R.T
    return R

def ComputeBayesian(X):
    data=bdgraph.bdgraph(X) #too slow otherwise
    bdprec=np.array(data[0])
    if bdprec is None:
        print 'Didnt converge?'
        bdprec=np.eye(X.shape[1])
    return bdprec

from sklearn import metrics

def ranking_precision_score(y_true, y_score, k=10):
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)
    return float(n_relevant) / min(n_pos, k)

def evalData2(PredAdjs,TrueAdjs,text='Data ',Latex=False,torep=37):
    diff=PredAdjs-TrueAdjs
    test_set_y=TrueAdjs
    z=PredAdjs
    Q=test_set_y.shape[0]
    Pk37_t=[]
    auc2=[]
    CEs=[]
    nan_trials=[]
    for i in range(Q):
        if(np.count_nonzero(np.isnan(PredAdjs[i]))):
            nan_trials.append(i)
            print(i)
        else:
            Pk37_t.append(ranking_precision_score(test_set_y[i], z[i], k=torep))
            fpr, tpr, thresholds = metrics.roc_curve(TrueAdjs[i],PredAdjs[i], pos_label=1)
            auc2.append(metrics.auc(fpr, tpr))
            CEs.append(metrics.mean_absolute_error(TrueAdjs[i],PredAdjs[i]))
    precision,recall,_=metrics.precision_recall_curve(TrueAdjs.ravel(),PredAdjs.ravel(),pos_label=1)
    cross=metrics.log_loss(TrueAdjs.ravel(),PredAdjs.ravel())
    
    fpr, tpr, thresholds = metrics.roc_curve(TrueAdjs.ravel(),PredAdjs.ravel(), pos_label=1)
    auc=metrics.auc(fpr, tpr)
    
    #print text,' AP',AP,'MSE',np.mean((diff)**2),'Cross-entropy:',cross
    
   
    
    
        
    Pk37=np.sum(Pk37_t)/Q
    
    CE=metrics.mean_absolute_error(TrueAdjs.ravel(),PredAdjs.ravel())
    if(Latex):
        Pk37ste=stats.sem(Pk37_t)
        aucSte=stats.sem(auc2)
        CESte=stats.sem(CEs)
        CEs=np.mean(CEs)
        print '& %s & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f  & %.3f $\pm$ %.3f \\\\'%(text,Pk37,Pk37ste,auc,aucSte,CEs,CESte)
    else:
        print '%s|AUC:%.2f|Prec@5%:%.2f|logloss:%.2f|MAE:%.2f'%(text,auc,Pk37,cross,CE)

    
