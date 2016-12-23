# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 15:56:58 2016

@author: joe
"""
from EvaluationTools import evalData2,toPartialCorr
import numpy as np
import os
os.environ['THEANO_FLAGS'] = "device=cpu"   
import theano
from keras.models import model_from_json
n_features=39
n_samp=35
alph=0.95


MODEL_PATH='models/'
model=model_from_json(open('%s/diag3_edge_features_%d_samples_%d_alpha_%d.json'%(MODEL_PATH,n_features,n_samp,alph*100)).read())
model.load_weights('%s/diag3_edge_features_%d_samples_%d_alpha_%d.h5'%(MODEL_PATH,n_features,n_samp,alph*100))


import pickle
graphType="random"
sparsity=0.05
Repetitions=5
n_samp=35
dist="Gaussian"




Truth=[]
TruthPartial=[]
GlassoSol=[]
GlassoSolBon=[]
NNSol=[]
NNSolP=[]


R_Install=False
glbon=False #best graph lasso soln
#compute Bdgraph solution .. need R installation 
bayesian=False
BDSol=[]


if(not R_Install):
    from GenSynthCov_New import generate_cov_learn_dataset_repeat,spd_to_vector,spd_to_vector_nondiag
    true_covariances,true_precisions,noised_covariances,sigs=generate_cov_learn_dataset_repeat(n_signals=n_samp,n_features=n_features,repeats=1,n_samples=Repetitions,alpha=alph,random_state=1)
        
    test_set_y=np.array([np.abs(spd_to_vector_nondiag(toPartialCorr(M))) for M in true_precisions])
    test_set_y[test_set_y!=0]=1    
    if(td):
        test_set_x=np.expand_dims(np.array([(M) for M in noised_covariances]),axis=1)
    else:
        test_set_x=np.array([spd_to_vector(M) for M in noised_covariances])
    test_set_sigs=sigs
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
        X=tes_set_sigs[rep][0:n_samp]
    
    TrueAdj=TruePrec.copy()
    #print TrueAdj
    TrueAdj[np.abs(TruePrec)<1e-7]=0
    TrueAdj[np.abs(TruePrec)>1e-7]=1

    #Apply the NN
    NNadj=applyNN(X,model)
    NNadjP=applyNNRandom(X,model,reps=20)
  
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
    
    GlassoSol.append(glprec[np.triu_indices(n_features,k=1)])
    NNSol.append(NNadj[np.triu_indices(n_features,k=1)])
    NNSolP.append(NNadjP[np.triu_indices(n_features,k=1)])
    
    
    sys.stdout.write("\r %.2f%% Complete, Sparsity %d" %(float(rep*100)/Repetitions,(np.sum(TrueAdj)-n_features)/2))
    
#evalData2(np.array(Zeros),np.array(Truth,dtype='int64'),text='Zeros',Latex=False)
evalData2(np.array(GlassoSol),np.array(Truth,dtype='int64'),text='Glasso',Latex=False)
evalData2(np.array(NNSol),np.array(Truth,dtype='int64'),text='DeepGraph-39',Latex=False)
evalData2(np.array(NNSolP),np.array(Truth,dtype='int64'),text='DeepGraph-39+Perm',Latex=False)
#ev.evalData2(np.array(GlassoSolBon),np.array(Truth,dtype='int64')[tokeep],text='Glasso (optimal)',Latex=False)
   