# -*- coding: utf-8 -*-
"""
Eugene Belilovsky
eugene.belilovsky@inria.fr
"""

Use_R=False
import sys
from sklearn.utils import check_random_state
from sklearn.datasets.samples_generator import make_sparse_spd_matrix
import numpy as np

if (Use_R):
    import rpy2
    import readline
    from rpy2.robjects.packages import importr
    bdgraph=importr("BDgraph")
    rags2ridges=importr("rags2ridges")
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
epsilon=1e-7

def generate_signal_from_covariance(covariance,samples=100,
                                     random_state=0):

    random_state = check_random_state(random_state)
    mean = np.zeros(covariance.shape[0])
    return random_state.multivariate_normal(mean,covariance,(samples,))

def generate_cov_learn_dataset_repeat_gwishart(n_signals=50,n_features=15,n_samples=100,alpha=0.95,repeats=10,random_state=0,graphType="random",verbose=True):
    
    true_covariances=[]
    true_precisions=[]
    noised_covariances=[]
    sigs=[]
    
    for i in range(n_samples):
        data=bdgraph.bdgraph_sim( n = 5, p = n_features, type = "Gaussian", graph = graphType, prob = 1.-alpha)
        Xall=np.array(data[1])
        cov=np.array(data[2])
        prec=np.array(data[3])
        prec[np.abs(prec)<epsilon]=0
        #TODO lets check this is right
        for j in range(repeats):
            X=generate_signal_from_covariance(cov,samples=n_signals,random_state=i+j+random_state+1)
            X -= X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1
            X /= std
            cov_emp=X.T.dot(X)/X.shape[0]
            
            true_covariances.append(cov)#careful if ever using this we normalized the covariance
            true_precisions.append(prec)#only the 0 pattern remains
            noised_covariances.append(cov_emp)
            sigs.append(X)
        if(verbose):
            sys.stdout.write("Generating Data \r%.2f%%" % (float(i*100)/n_samples))
            sys.stdout.flush()
    return true_covariances,true_precisions,noised_covariances,sigs


def generate_cov_learn_dataset_repeat(n_signals=50,n_features=15,n_samples=100,alpha=0.95,
                                      repeats=10,random_state=0,verbose=True,normalize=False,
                                      laplace=False,permute_repeats=False,graphType="random",mix_with_random=False,smallest_coef=-.9):
    true_covariances=[]
    true_precisions=[]
    noised_covariances=[]
    sigs=[]
    I=np.eye(n_features)
    ind=np.arange(0,n_features)
    ind2=ind.copy()
    for i in range(n_samples):
        if(graphType=='smallWorld'):
            if(mix_with_random and np.random.rand(1)[0]<0.5):
                prec=make_sparse_spd_matrix(n_features,alpha=alpha,smallest_coef=smallest_coef,random_state=i+random_state)
            else:
                data=rags2ridges.createS(10,n_features,topology="small-world",precision=True)
                prec=np.array(data)
                np.random.shuffle(ind2)
                I=np.eye(prec.shape[1])
                P=I[:,ind2]
                C=np.zeros((prec.shape[1],prec.shape[1]))
                C[np.triu_indices(n_features,k=1)]=prec[np.triu_indices(n_features,k=1)]
                C=C+C.T
                C=P.dot(C).dot(P.T)
                prec=C+I
        else:
            prec=make_sparse_spd_matrix(n_features,alpha=alpha,smallest_coef=smallest_coef,random_state=i+random_state)
        
        cov=np.linalg.inv(prec)
        for j in range(repeats):
            if(laplace):
                # see prop 3.1 in "A multivariate generalization of the power exponential family of distributions" E. Gomez et al 1998  
                E=np.tile(np.random.exponential(scale=1.0, size=n_signals),(n_features,1)).T
                Z=np.random.multivariate_normal(np.zeros(n_features),cov,n_signals)
                X=(E)**(0.5)*Z
            else:
                X=generate_signal_from_covariance(cov,samples=n_signals,random_state=i+j+random_state+1)
        
            if(normalize):
                X -= X.mean(axis=0)
                std = X.std(axis=0)
                std[std == 0] = 1
                X /= std
            cov_emp=X.T.dot(X)/X.shape[0]
            if(permute_repeats):
                #This seems to mess up
                np.random.shuffle(ind)
                P=I[:,ind]
                cov_emp=P.T.dot(cov_emp).dot(P)
                true_covariances.append(P.dot(cov).dot(P.T))
                true_precisions.append(P.dot(prec).dot(P.T))
            else:
                true_covariances.append(cov)
                true_precisions.append(prec)
            noised_covariances.append(cov_emp)
            sigs.append(X)
        if(verbose):
            sys.stdout.write("\r%.2f%%" % (float(i*100)/n_samples))
            sys.stdout.flush()
    return true_covariances,true_precisions,noised_covariances,sigs

def corrupt_cov(cov_true,n_signals=50):
    X=generate_signal_from_covariance(cov_true,samples=n_signals,random_state=None)
    return X.T.dot(X)/n_signals
def spd_to_vector(M):
    return M[np.triu_indices(M.shape[0])]

def spd_to_vector_nondiag(M,scale=False):
    result=M[np.triu_indices(M.shape[0],k=1)]
    if(scale):
        result=np.abs(result)/np.max(np.abs(result))
    return result    
    
def generateFromCovs(true_covariances,n_signals=50):
    return [corrupt_cov(cov_true,n_signals=n_signals) for cov_true in true_covariances]
