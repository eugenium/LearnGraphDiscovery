# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:56:58 2016

@author: joe
"""
from sklearn.covariance import GraphLassoCV
from EvaluationTools import evalData2,toPartialCorr
import EvaluationTools as ev
import numpy as np
import os,sys
os.environ['THEANO_FLAGS'] = "device=cpu"   
import theano
from keras.models import model_from_json
import pickle
graphType="random"
sparsity=0.05
Repetitions=25
n_features=39
alph=0.95
n_samp=35


MODEL_PATH='models/'
model=model_from_json(open('%s/diag3_edge_features_%d_samples_%d_alpha_%d.json'%(MODEL_PATH,n_features,n_samp,alph*100)).read())
model.load_weights('%s/diag3_edge_features_%d_samples_%d_alpha_%d.h5'%(MODEL_PATH,n_features,n_samp,alph*100))








Truth=[]
TruthPartial=[]
GlassoSol=[]
GlassoSolBon=[]
NNSolP=[]
Random=[]


R_Install=False
glbon=False #best graph lasso soln
#compute Bdgraph solution .. need R installation 
bayesian=False
BDSol=[]


if(not R_Install):
    from GenSynthCov_New import generate_cov_learn_dataset_repeat,spd_to_vector,spd_to_vector_nondiag
    true_covariances,true_precisions,noised_covariances,sigs=generate_cov_learn_dataset_repeat(n_signals=n_samp,n_features=n_features,repeats=1,n_samples=Repetitions,alpha=alph,random_state=1)

for rep in xrange(Repetitions):
    # random synthetic data
    if(R_Install):
        #install R and bdgraph, and rpy2 to run these simulations
        data=bdgraph.bdgraph_sim( n = n_samp, p = n_features, type = "Gaussian", 
                                 graph = graph, prob = sparsity)
        TruSigma=np.array(data[2])
        TruePrec=np.array(data[3])
        X=np.array(data[1])[0:n_samp]
    else:
        TruSigma=true_covariances[rep]
        TruePrec=true_precisions[rep]
        X=sigs[rep][0:n_samp]
    
    TrueAdj=TruePrec.copy()
    #print TrueAdj
    TrueAdj[np.abs(TruePrec)<1e-7]=0
    TrueAdj[np.abs(TruePrec)>1e-7]=1

    #Apply the NN
    NNadjP=ev.applyNNRandom(X,model,reps=20)
  
    #Apply the Glasso
    gl=GraphLassoCV().fit(X)
    glprec=gl.precision_.copy()
    glprec=np.abs(toPartialCorr(glprec))
    glprec[np.diag_indices(glprec.shape[0])]=0


    if(glbon):
        glbonprec=glassoBonaFidePartial(gl,X,TruSigma)
        glbonprec[np.diag_indices(glprec.shape[0])]=0    
        GlassoSolBon.append(glbonprec[np.triu_indices(n_features,k=1)])
    if(bayesian):
        #BDgraph solver
        data=bdgraph.bdgraph(X[0:n_samps])
        bdprec=np.array(data[0])
        BDSol.append(bdprec[np.triu_indices(n_features,k=1)])
    
    
    Truth.append(TrueAdj[np.triu_indices(n_features,k=1)])
    Partial=toPartialCorr(TruePrec)
    TruthPartial.append(Partial[np.triu_indices(n_features,k=1)])
    Random.append(ev.RandAdjc(X)[np.triu_indices(n_features,k=1)])
    
    GlassoSol.append(glprec[np.triu_indices(n_features,k=1)])
    NNSolP.append(NNadjP[np.triu_indices(n_features,k=1)])
    
    
    sys.stdout.write("\r %.2f%% Complete, Sparsity %d" %(float(rep*100)/Repetitions,(np.sum(TrueAdj)-n_features)/2))
    
print '\n'
evalData2(np.array(Random),np.array(Truth,dtype='int64'),text='Random',Latex=False)
evalData2(np.array(GlassoSol),np.array(Truth,dtype='int64'),text='Glasso',Latex=False)
evalData2(np.array(NNSolP),np.array(Truth,dtype='int64'),text='DeepGraph',Latex=False)
#ev.evalData2(np.array(GlassoSolBon),np.array(Truth,dtype='int64')[tokeep],text='Glasso (optimal)',Latex=False)
   