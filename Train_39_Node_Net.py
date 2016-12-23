"""
Learn to decode a noised covariance matrix

"""

import time

import numpy as np

from sklearn import metrics
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input,Flatten, merge
from keras.layers.convolutional import Convolution2D,AtrousConvolution2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import activity_l1,activity_l2
from keras.callbacks import ModelCheckpoint
import keras.callbacks
from keras import backend as K
from GenSynthCov_New import generate_cov_learn_dataset_repeat,spd_to_vector,spd_to_vector_nondiag

CheckPoint_PATH='/tmp/' 
FINAL_PATH='models/'
K.set_image_dim_ordering('th')
n_features=39
alph=0.95
n_samp=35


def GetDiag(var):
    from theano.tensor.nnet.conv3d2d import DiagonalSubtensor
    takeDiag = DiagonalSubtensor()
    [s1,s2,s3,s4]=var.shape
    diag=takeDiag(var,2,3)
    a=diag.reshape((s1,s2,1,s3)).repeat(s3,axis=2)
    b=diag.reshape((s1,s2,s3,1)).repeat(s3,axis=3)
    return a+b
def out_diag_shape(input_shape):
    return input_shape
def constructNet(input_dim=784,n_hidden=1000,n_out=1000,nb_filter=50,prob=0.5,lr=0.0001):
    nb_filters=50
    input_img= Input(shape=list(input_dim))
    a = input_img

    a1 = AtrousConvolution2D(nb_filters, 3, 3,atrous_rate=(1,1),border_mode='same')(a)    
    b = AtrousConvolution2D(nb_filters, 3, 3,atrous_rate=(1,1),border_mode='same')(a)  #We only use the diagonal output from this, TODO: only filter diagonal
    a2=Lambda(GetDiag, output_shape=out_diag_shape)(b)
    comb=merge([a1,a2],mode='sum')
    comb = BatchNormalization()(comb)  
    a = Activation('relu')(comb)
    
    l=5
    for i in range(1,l):
        a1 = AtrousConvolution2D(nb_filters, 3, 3,atrous_rate=(l,l),border_mode='same')(a)    
        b = AtrousConvolution2D(nb_filters, 3, 3,atrous_rate=(l,l),border_mode='same')(a)  #We only use the diagonal output from this, TODO: only filter diagonal
        a2=Lambda(GetDiag, output_shape=out_diag_shape)(b)
        comb=merge([a1,a2],mode='sum')
        comb = BatchNormalization()(comb)  
        a = Activation('relu')(comb)
        
    decoded = Convolution2D(1, 1, 1, activation='sigmoid', border_mode='same')(a)
    final=Flatten()(decoded)
    model = Model(input_img, final)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def ranking_precision_score(y_true, y_score, k=10):
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)
def evalModel(model,test_set_x,test_set_y,m):
    z=model.predict(test_set_x)
    z=z.reshape(-1,m,m)
    z=np.array([spd_to_vector_nondiag(M) for M in z])
    return evalData(z,test_set_y)
    
def evalData(z,test_set_y):
    " z- prediction test_set_y is the truth "
    diff=z-test_set_y
    fpr, tpr, thresholds = metrics.roc_curve(test_set_y.ravel(), z.ravel(), pos_label=1)
    auc=metrics.auc(fpr, tpr)
    ap=metrics.average_precision_score(test_set_y.ravel(), z.ravel())
    
    Q=test_set_y.shape[0]
    Pk10=0
    Pk20=0
    Pk30=0
    Pk50=0
    Pk37=0
    for i in range(Q):
        Pk10+=ranking_precision_score(test_set_y[i], z[i], k=10)
        Pk20+=ranking_precision_score(test_set_y[i], z[i], k=20)
        Pk30+=ranking_precision_score(test_set_y[i], z[i], k=30)
        Pk37+=ranking_precision_score(test_set_y[i], z[i], k=37)
        Pk50+=ranking_precision_score(test_set_y[i], z[i], k=30)
    Pk10=Pk10/Q
    Pk20=Pk20/Q
    Pk30=Pk30/Q
    Pk50=Pk50/Q
    Pk37=Pk37/Q
    cross=metrics.log_loss(test_set_y,z)
    print '\n'
    print 'AUC',auc,'MSE',np.mean((diff)**2),'Cross-entropy:',cross
    print 'Precision at k=10: ',Pk10,' k=20: ',Pk20,' k=30: ',Pk30,' k=50: ',Pk50, ' k=37: ',Pk37
    return Pk37

def toPartialCorr(Prec):
    D=Prec[np.diag_indices(Prec.shape[0])]
    P=Prec.copy()
    D=np.outer(D,D)
    return -P/np.sqrt(D)
    
def datagenerate2(n_samp,n_features,alph,trainset=10000,repeats=10,testset=50,random_state=0,td=True):
    true_covariances,true_precisions,noised_covariances,sigs=generate_cov_learn_dataset_repeat(n_signals=n_samp,n_features=n_features,repeats=repeats,n_samples=trainset,alpha=alph,normalize=True,random_state=random_state)
        
    train_set_y=np.expand_dims(np.array([np.abs(spd_remove_diag(toPartialCorr(M))) for M in true_precisions]),axis=1)
   # train_set_y[train_set_y!=0]=1

    if(td):
        train_set_x=np.expand_dims(np.array([spd_remove_diag(M) for M in noised_covariances]),axis=1)
    else:
        train_set_x=np.array([spd_to_vector(M) for M in noised_covariances])

    true_covariances,true_precisions,noised_covariances,sigs=generate_cov_learn_dataset_repeat(n_signals=n_samp,n_features=n_features,repeats=1,n_samples=testset,alpha=alph,normalize=True,random_state=123456+trainset)
        
    test_set_y=np.expand_dims(np.array([np.abs(spd_to_vector_nondiag(toPartialCorr(M))) for M in true_precisions]),axis=1)
    test_set_y[test_set_y!=0]=1    
    if(td):
        test_set_x=np.expand_dims(np.array([spd_remove_diag(M) for M in noised_covariances]),axis=1)
    else:
        test_set_x=np.array([spd_to_vector(M) for M in noised_covariances])
    test_set_sigs=sigs
    
    return train_set_x,train_set_y,test_set_x,test_set_y,test_set_sigs
def toPartialCorr(Prec):
    D=Prec[np.diag_indices(Prec.shape[0])]
    P=Prec.copy()
    D=np.outer(D,D)
    return -P/np.sqrt(D)
    
if __name__ == "__main__":
    learning_rate=0.0001
    training_epochs=35
    epochs=10
    batch_size=256
    prob=0.5
    rng = np.random.RandomState(123)
    
    
    #create dummy set for initilizing the network
    train_set_x,train_set_y,test_set_x,test_set_y,test_set_sigs=datagenerate(n_samp,n_features,alph,trainset=2,repeats=1,testset=1,random_state=1)
    model=constructNet(input_dim=train_set_x.shape[1:],n_hidden=50,n_out=train_set_y.shape[1],prob=0.5,lr=learning_rate)
    json_string = model.to_json()
    open('%s/diag3_edge_features_%d_samples_%d_alpha_%d.json'%(CheckPoint_PATH,n_features,n_samp,alph*100), 'w').write(json_string)
    import pickle
    
    
#    
    train_set_x,train_set_y,test_set_x,test_set_y,test_set_sigs=datagenerate(n_samp,n_features,alph,trainset=500,repeats=5,testset=50,random_state=1)
    pickle.dump([test_set_x,test_set_y],open('%s/validation_%d_samples_%d_alpha_%d.pkl'%(CheckPoint_PATH,n_features,n_samp,alph*100), 'wb'))
        ###
    covs=[]
    covEmp=[]
    
    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            evalModel(self.model,test_set_x,test_set_y,n_features)
    ls=LossHistory()
    start_time = time.clock()
    evalModel(model,test_set_x,test_set_y,n_features)
    checkpointer = ModelCheckpoint(filepath='%s/diag3_edge_features_%d_samples_%d_alpha_%d.h5'%(CheckPoint_PATH,n_features,n_samp,alph*100), verbose=1, save_best_only=True)
    for i in range(training_epochs):
        ea=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        model.fit(train_set_x, train_set_y, nb_epoch=epochs,validation_split=0.1,callbacks=[ls,ea,checkpointer],batch_size=batch_size)
        evalModel(model,test_set_x,test_set_y,n_features)
        train_set_x,train_set_y,_,_,_=datagenerate(n_samp,n_features,alph,trainset=5000,repeats=3,testset=2,random_state=(i+2)*1000000)
    end_time = time.clock()
    training_time = (end_time - start_time)
    print 'training time',training_time
   
    
    json_string = model.to_json()
    open('%s/diag3_edge_features_%d_samples_%d_alpha_%d.json'%(FINAL_PATH,n_features,n_samp,alph*100), 'w').write(json_string)
    model.save_weights('%s/diag3_edge_features_%d_samples_%d_alpha_%d.h5'%(FINAL_PATH,n_features,n_samp,alph*100))
   

