from sklearn.covariance.empirical_covariance_ import log_likelihood,fast_logdet
from sklearn.covariance import LedoitWolf,GraphLassoCV,EmpiricalCovariance,graph_lasso
from sklearn.covariance.graph_lasso_ import graph_lasso_path
from ips.ips import prop_scaling
import numpy as np

from scipy import stats
evalonly=False
if(not evalonly):
    import rpy2
    import readline
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    bdgraph=importr("BDgraph")
    rags2ridges=importr("rags2ridges")
    
    from sklearn.covariance import GraphLassoCV,LedoitWolf
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
def ComputeResults_OneSubset(X,Xtest,methods,methodHandles,edges,verbose=False,normTest=False):
    n_features=X.shape[1]
    results=dict()
    adjacency=[]
    for func in methodHandles:
        adjacency.append(func(X))

    for method in methods:
        results[method]=[]

    if(normTest):
        Xtest -= Xtest.mean(axis=0)
        std = Xtest.std(axis=0)
        std[std == 0] = 1
        Xtest /= std
    emp_cov=Xtest.T.dot(Xtest)/Xtest.shape[0]
    
    #Standardize the data before computing the empirical covariance
    #IPS doesnt do any data processing
    X -= X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1
    X /= std
    emp_cov_train=X.T.dot(X)/X.shape[0]
#    emp_cov_train=EmpiricalCovariance().fit(X).covariance_
#    emp_cov=EmpiricalCovariance().fit(Xtest).covariance_
    
    for num_edges in edges:
        #select edges
        for M,method in zip(adjacency, methods):
            supp=np.abs(M)
            supp[np.diag_indices(supp.shape[0])]=0
            vals=np.sort(supp[np.triu_indices(n_features,k=1)])[::-1]
            thresh=max(np.min(vals[vals>0]),vals[num_edges-1])
         #   print method,' Threshold',thresh,vals[num_edges]    
            supp[supp<thresh]=0
            supp[supp>0]=1
            
            if(np.sum(supp)/2>num_edges):
                print 'Ties correcting'
                flat=supp[np.triu_indices(n_features,k=1)]
                ind=np.where(flat!=0)[0]
                s=np.zeros((n_features,n_features))
                flat=s[np.triu_indices(n_features,k=1)]
                flat[ind[0:num_edges]]=1
                s[np.triu_indices(n_features,k=1)]=flat
                supp=s+s.T
            supp[np.diag_indices(supp.shape[0])]=1
            prec=prop_scaling(emp_cov_train, supp)
            results[method].append(log_likelihood(emp_cov,prec))
            if(verbose):
                print method,thresh,num_edges,'prec',Xtest[0,1:5],' ll:',log_likelihood(emp_cov,prec),(np.sum(supp!=0)-n_features)/2,(np.sum(np.abs(prec)>1e-7)-n_features)/2
               # print vals[0:15]
    return results,adjacency
    

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
    Pk10=0
    Pk20=0
    Pk30=0
    Pk37_t=[]
    Pk50=0
    APs=[]
    auc2=[]
    CEs=[]
    nan_trials=[]
    for i in range(Q):
        if(np.count_nonzero(np.isnan(PredAdjs[i]))):
            nan_trials.append(i)
            print(i)
        else:
            Pk10+=ranking_precision_score(test_set_y[i], z[i], k=10)
            Pk20+=ranking_precision_score(test_set_y[i], z[i], k=20)
            Pk30+=ranking_precision_score(test_set_y[i], z[i], k=30)
            Pk37_t.append(ranking_precision_score(test_set_y[i], z[i], k=torep))
            Pk50+=ranking_precision_score(test_set_y[i], z[i], k=50)
            fpr, tpr, thresholds = metrics.roc_curve(TrueAdjs[i],PredAdjs[i], pos_label=1)
            auc2.append(metrics.auc(fpr, tpr))
            CEs.append(metrics.mean_absolute_error(TrueAdjs[i],PredAdjs[i]))
            APs.append(metrics.average_precision_score(TrueAdjs[i].ravel(),PredAdjs[i].ravel()))
    precision,recall,_=metrics.precision_recall_curve(TrueAdjs.ravel(),PredAdjs.ravel(),pos_label=1)
    AP=metrics.average_precision_score(TrueAdjs.ravel(),PredAdjs.ravel())#metrics.fbeta_score(TrueAdjs.ravel(),PredAdjs.ravel(), beta=1,pos_label=1)
    cross=metrics.log_loss(TrueAdjs.ravel(),PredAdjs.ravel())
    
    fpr, tpr, thresholds = metrics.roc_curve(TrueAdjs.ravel(),PredAdjs.ravel(), pos_label=1)
    auc=metrics.auc(fpr, tpr)
    
    #print text,' AP',AP,'MSE',np.mean((diff)**2),'Cross-entropy:',cross
    
   
    
    
        
    Pk10=Pk10/Q
    Pk20=Pk20/Q
    Pk30=Pk30/Q
    Pk37=np.sum(Pk37_t)/Q
    Pk50=Pk50/Q
    
    CE=metrics.mean_absolute_error(TrueAdjs.ravel(),PredAdjs.ravel())
    if(Latex):
        Pk37ste=stats.sem(Pk37_t)
        aucSte=stats.sem(auc2)
        CESte=stats.sem(CEs)
        APste=stats.sem(APs)
        APs=np.mean(APs)
        CEs=np.mean(CEs)
        #print Pk37_t
       # print '%s & %.3f & %.3f & %.2f & %.2f & %.3f & %.2f'%(text,cross,auc,Pk20,Pk37,AP,CE)
        #print '& %s & %.3f $\pm$ %.3f &%.3f $\pm$ %.3f& %.3f $\pm$ %.3f  & %.2f $\pm$ %.2f \\\\'%(text,Pk37,2*Pk37ste,APs,APste,auc,2*aucSte,CEs,2*CESte)
        print '& %s & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f  & %.3f $\pm$ %.3f \\\\'%(text,Pk37,Pk37ste,auc,aucSte,CEs,CESte)
    else:
        print '%s|AP:%.3f|AUC:%.2f|Prec@k=(10,20,30,37,6237)=(%.2f,%.2f,%.2f,%.2f,%.2f)|logloss:%.2f|MAE:%.2f'%(text,
                                                                                                     AP,auc,Pk10,
                                                                                                     Pk20,Pk30,
                                                                                                     Pk37,Pk50,
                                                                                                     cross,CE)
    #print 'Precision at k=10:',Pk10,' k=20:',Pk20,' k=30',Pk30,' k=50:',Pk50
    #print 'Calibration Error',CE
    
    
   # TotalEdges=(TrueAdjs[0].shape[0])*(TrueAdjs[0].shape[0]-1)/2
    
   # max_found=max(np.sum(TrueAdjs>0,axis=1)
    data=dict()
    data['precision']=precision
    data['recall']=recall
    data['AP']=AP
    data['Cross']=cross
    data['CE']=CE
    data['fpr']=fpr
    data['tpr']=tpr
    data['auc']=auc
    data['nan']=nan_trials
    return data
    

import sys
#def ComputeSyntheticExampe_Parallel(model,model2,graphType="random",sparsity=0.05,Repetitions=10,
#                           bayesian=False,perm=False,n_samps=35,dist="Gaussian",n_features=39,glbon=False):
#    Truth=[]
#    TruthPartial=[]
#    GlassoSol=[]
#    GlassoSolBon=[]
#    NNSol=[]
#    NNSol2=[]
#    NNSolP=[]
#    NNSolP2=[]
#    BDSol=[]
#    
#    
#    for rep in xrange(Repetitions):
#    def oneLoop():   
#        if(graphType=="small"):
#            #G=nx.watts_strogatz_graph(n_features,3,0.5)#navigable_small_world_graph(10)
#           # G=nx.scale_free_graph(n_features)
#            data=rags2ridges.createS(10,n_features,topology="small-world",precision=True)
#            TruePrec=np.array(data)
#            TruSigma=np.linalg.inv(TruePrec)
#            X=generate_signal_from_covariance(TruSigma,samples=n_samps,random_state=None)
#            #graph="hub"
#        else:
#            graph=graphType
#            data=bdgraph.bdgraph_sim( n = n_samps, p = n_features, type = "Gaussian", 
#                                     graph = graph, prob = sparsity)
#            TruSigma=np.array(data[2])
#            TruePrec=np.array(data[3])
#        
#      # print min(np.linalg.eigvalsh(TruSigma)),min(np.linalg.eigvalsh(TruePrec))
#        if(dist=="Laplace"):
#            print 'Using Laplace'
#            symmetricCov=(TruSigma+TruSigma.T)/2 #make sure its symmetric to finite precision - laplacedemon requires 
#            if(np.sum(np.abs(symmetricCov-TruSigma))>1e-7):#this should be zero basically
#                print 'Something went wrong with the Sigma symmetry'
#            X=Gen_Laplace(n_samps,symmetricCov)
#        else:
#            if(graphType!="small"):
#                X=np.array(data[1])
#            
#        TrueAdj=TruePrec.copy()
#        #print TrueAdj
#        TrueAdj[np.abs(TruePrec)<1e-9]=0
#        TrueAdj[np.abs(TruePrec)>1e-9]=1
#
#   
#        NNadj=applyNN(X[0:n_samps],model)
#        NNadj2=applyNN(X[0:n_samps],model2)
#        gl=GraphLassoCV().fit(X[0:n_samps])
#        glprec=gl.precision_.copy()
#        glprec=np.abs(toPartialCorr(glprec))
#        glprec[np.diag_indices(glprec.shape[0])]=0
#
#
#        if(glbon):
#            glbonprec=glassoBonaFidePartial(gl,X,TruSigma)
#            glbonprec[np.diag_indices(glprec.shape[0])]=0
#        else:
#            glbonprec=None
#        if(bayesian):
#            #BDgraph solver
#            data=bdgraph.bdgraph(X[0:n_samps])
#            bdprec=np.array(data[0])
#        else:
#            bdprec=glprec
#           
#        Truth.append(TrueAdj[np.triu_indices(n_features,k=1)])
#        Partial=toPartialCorr(TruePrec)
#        TruthPartial.append(Partial[np.triu_indices(n_features,k=1)])
#        
#        GlassoSol.append(glprec[np.triu_indices(n_features,k=1)])
#        GlassoSolBon.append(glbonprec[np.triu_indices(n_features,k=1)])
#        NNSol.append(NNadj[np.triu_indices(n_features,k=1)])
#        NNSol2.append(NNadj2[np.triu_indices(n_features,k=1)])
#        BDSol.append(bdprec[np.triu_indices(n_features,k=1)])
#        
#        if(perm):
#            NNadjP=applyNNRandom(X[0:n_samps],model,reps=20)
#            NNadjP2=applyNNRandom(X[0:n_samps],model2,reps=20)
#            NNSolP.append(NNadjP[np.triu_indices(n_features,k=1)])
#            NNSolP2.append(NNadjP2[np.triu_indices(n_features,k=1)])
#        return Truth,TruthPartial,GlassoSol,NNSol,NNSol2,BDSol,NNSolP,NNSolP2,GlassoSolBon 
#    
#    return Truth,TruthPartial,GlassoSol,NNSol,NNSol2,BDSol,NNSolP,NNSolP2,GlassoSolBon 
    
    
def ComputeSyntheticExampe(model,model2,graphType="random",sparsity=0.05,Repetitions=10,
                           bayesian=False,perm=False,n_samps=35,dist="Gaussian",n_features=39,glbon=False,onlyNN=False):
    Truth=[]
    TruthPartial=[]
    GlassoSol=[]
    GlassoSolBon=[]
    NNSol=[]
    NNSol2=[]
    NNSolP=[]
    NNSolP2=[]
    BDSol=[]
    
    for rep in xrange(Repetitions):
        # random synthetic data
        if(graphType=="small"):
            #G=nx.watts_strogatz_graph(n_features,3,0.5)#navigable_small_world_graph(10)
           # G=nx.scale_free_graph(n_features)
            data=rags2ridges.createS(10,n_features,topology="small-world",precision=True)
            TruePrec=np.array(data)
            TruSigma=np.linalg.inv(TruePrec)
            X=generate_signal_from_covariance(TruSigma,samples=n_samps,random_state=None)
            #graph="hub"
        else:
            graph=graphType
            data=bdgraph.bdgraph_sim( n = n_samps, p = n_features, type = "Gaussian", 
                                     graph = graph, prob = sparsity)
            TruSigma=np.array(data[2])
            TruePrec=np.array(data[3])
        
      # print min(np.linalg.eigvalsh(TruSigma)),min(np.linalg.eigvalsh(TruePrec))
        if(dist=="Laplace"):
            print 'Using Laplace'
            symmetricCov=(TruSigma+TruSigma.T)/2 #make sure its symmetric to finite precision - laplacedemon requires 
            if(np.sum(np.abs(symmetricCov-TruSigma))>1e-7):#this should be zero basically
                print 'Something went wrong with the Sigma symmetry'
            X=Gen_Laplace(n_samps,symmetricCov)
        else:
            if(graphType!="small"):
                X=np.array(data[1])
            
        TrueAdj=TruePrec.copy()
        #print TrueAdj
        TrueAdj[np.abs(TruePrec)<1e-7]=0
        TrueAdj[np.abs(TruePrec)>1e-7]=1

   
        NNadj=applyNN(X[0:n_samps],model)
        NNadj2=applyNN(X[0:n_samps],model2)
        if(not onlyNN):
            gl=GraphLassoCV().fit(X[0:n_samps])
            glprec=gl.precision_.copy()
            glprec=np.abs(toPartialCorr(glprec))
            glprec[np.diag_indices(glprec.shape[0])]=0
        else:
            glprec=NNadj

        if(glbon):
            glbonprec=glassoBonaFidePartial(gl,X,TruSigma)
            glbonprec[np.diag_indices(glprec.shape[0])]=0
        else:
            glbonprec=glprec
        if(bayesian):
            #BDgraph solver
            data=bdgraph.bdgraph(X[0:n_samps])
            bdprec=np.array(data[0])
        else:
            bdprec=glprec

        Truth.append(TrueAdj[np.triu_indices(n_features,k=1)])
        Partial=toPartialCorr(TruePrec)
        TruthPartial.append(Partial[np.triu_indices(n_features,k=1)])
        
        GlassoSol.append(glprec[np.triu_indices(n_features,k=1)])
        GlassoSolBon.append(glbonprec[np.triu_indices(n_features,k=1)])
        NNSol.append(NNadj[np.triu_indices(n_features,k=1)])
        NNSol2.append(NNadj2[np.triu_indices(n_features,k=1)])
        BDSol.append(bdprec[np.triu_indices(n_features,k=1)])
        
        if(perm):
            NNadjP=applyNNRandom(X[0:n_samps],model,reps=20)
            NNadjP2=applyNNRandom(X[0:n_samps],model2,reps=20)
            NNSolP.append(NNadjP[np.triu_indices(n_features,k=1)])
            NNSolP2.append(NNadjP2[np.triu_indices(n_features,k=1)])
        sys.stdout.write("\r %.2f%% Complete, Sparsity %d" % 
                         (float(rep*100)/Repetitions,(np.sum(TrueAdj)-n_features)/2))
        sys.stdout.flush()
    print '\n'
    return Truth,TruthPartial,GlassoSol,NNSol,NNSol2,BDSol,NNSolP,NNSolP2,GlassoSolBon

if __name__ == "__main__":
    import os    
    os.environ['THEANO_FLAGS'] = "device=cpu"   
    import theano
    from keras.models import model_from_json
    n_features=39
    n_samp=35
    alph=0.95
    
    model2=model_from_json(open('savedNets/laplace_edge_features_%d_samples_%d_alpha_%d.json'%(n_features,n_samp,alph*100)).read())
    model2.load_weights('savedNets/laplace_edge_features_%d_samples_%d_alpha_%d.h5'%(n_features,n_samp,alph*100))
    model=model2
    #n_features=39
    #n_samp=35
    #alph=0.95
    #model=model_from_json(open('savedNets/loss2_new_edge_features_%d_samples_%d_alpha_%d.json'%(n_features,n_samp,alph*100)).read())
    #model.load_weights('savedNets/loss2_new_edge_features_%d_samples_%d_alpha_%d.h5'%(n_features,n_samp,alph*100))
    #print 'loaded NN model'
    import pickle
    graphType="random"
    sparsity=0.05
    Repetitions=20
    n_samp=35
    dist="Gaussian"
    bayesian=False
    Truth,TruthPartial,GlassoSol,NNSol,NNSol2,BDSol,NNSolP,GlassoSolBon=ComputeSyntheticExampe(model,model2,graphType=graphType,
                                                                                   sparsity=sparsity,
                                                                                   n_samps=n_samp,
                                                                                   dist=dist,
                                                                                   Repetitions=Repetitions,
                                                                                   bayesian=False,
                                                                                   perm=True,glbon=True
                                                                                  )
    pickle.dump([Truth,TruthPartial,GlassoSol,NNSol,NNSol2,BDSol,NNSolP],open(
                'samp_%d_%s_%s_sparsity_%d_reps_%d_bayesian_%d.pkl'%(n_samp,
                                                                     graphType,dist,100*(1-sparsity),
                                                                     Repetitions,int(bayesian)),'wb'))