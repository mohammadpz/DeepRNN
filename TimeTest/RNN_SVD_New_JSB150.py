#!/usr/bin/env python
import os
import gc

import sys
import glob
import  time
import numpy as np
np.set_printoptions(threshold=np.nan)
from numpy.random import uniform
from scipy import sparse
import scipy
import random
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
import logging
import time
import os
import datetime
import cPickle as pickle
from midi.utils import midiread, midiwrite
logger = logging.getLogger(__name__)
import math
import pickle 
mode = theano.Mode(linker='cvm')

#-----Definizione datasets-----#

trainingSet=glob.glob('../../data/JSB Chorales/train/*.mid')
testSet=glob.glob('../../data/JSB Chorales/test/*.mid')
validSet=glob.glob('../../data/JSB Chorales/valid/*.mid')

#trainingSet=glob.glob('../data/MuseData/train/*.mid')
#testSet=glob.glob('../data/MuseData/test/*.mid')
#testSet=glob.glob('../data/MuseData/valid/*.mid')

#trainingSet=glob.glob('../data/Nottingham/training_debug/*.mid')
#testSet=glob.glob('../data/Nottingham/test_debug/*.mid')
#validSet=glob.glob('../data/Nottingham/valid_debug/*.mid')

#trainingSet=glob.glob('../data/Piano-midi.de/train/*.mid')
#testSet=glob.glob('../data/Piano-midi.de/test/*.mid')
#testSet=glob.glob('../data/Piano-midi.de/valid/*.mid')




def LoadDataForPreTraining(r=(21, 109), dt=0.3):

    assert len(trainingSet) > 0, 'Training set is empty!' \
                           ' (did you download the data files?)'
    sampleLen=[]
    dataset=[]
    maxLen=0
    nSample=0
    for f in trainingSet:
        currentMidi=midiread(f, (21, 109),0.3).piano_roll#.astype(int)
        dataset.append(currentMidi)
        sampleLen.append(currentMidi.shape[0])
        if maxLen< currentMidi.shape[0]:
            maxLen=currentMidi.shape[0]
        nSample=nSample+1
    return (dataset, sampleLen, nSample, maxLen)
  
#------creazione sistema lineare per soluzione autoencoders-----#

def CreateMatrixForSvd(dataset=[],sampleLen=0,  nSample=0, maxLen=0, singleSampleDim=88):
  firstRowOfCurrentSeq=0
  rows=[]
  cols=[]
  data=[]
  
  nOfparsedSong=0
  for song in dataset:

    nOfparsedInstant=0
    for instant in song:

      for singleNote in range(0,singleSampleDim):
	if instant[singleNote]==1:
	  #lo inserisco tante volte quante le suo occorenze nella matrice M finale, che sono lungheza del brano- indice dell'istante
	  for offset in range(0,sampleLen[nOfparsedSong]-nOfparsedInstant):
	    #print "parsedsong:",nOfparsedSong
	    #print "parsed Instant",nOfparsedInstant
	    #print"-------------------"
	    #print "col:",(3*offset)+singleNote
	    cols.append((singleSampleDim*offset)+singleNote)
	    #print "row:",sum(sampleLen[0:nOfparsedSong])+offset+nOfparsedInstant
	    #print "------------------"
	    rows.append(sum(sampleLen[0:nOfparsedSong])+offset+nOfparsedInstant)
	    data.append(1)
      nOfparsedInstant=nOfparsedInstant+1
    nOfparsedSong=nOfparsedSong+1
  m=sparse.csc_matrix((data,(rows,cols)), shape=(sum(sampleLen),maxLen*singleSampleDim))
  return m
  

#------Metodi per calcolo SVD tramite matrici di covarianza e kernel----#

def SvdForBigData(m,singleSampleDim=88,nHidden=10):
  nSlice=(m.shape[1]/singleSampleDim)
  
  StartSlice=nHidden/88
  if nHidden%88 > 0:
    StartSlice=StartSlice+ 1
  
  
    
  
  #compute svd end X for the last slice, and then repeat this process for eache slice by joint with this one
  lastSlice=m[:, (nSlice-StartSlice)*singleSampleDim:nSlice*singleSampleDim]
  v,s,u_t=KeCSVD(lastSlice,nHidden)
  nSlice=nSlice-StartSlice
  for i in reversed(range(nSlice)):
    print "slice",i," of ",nSlice
    currentSVD=v*s
    currentSlice=m[:, i*singleSampleDim:i*singleSampleDim+singleSampleDim]
    #print currentSlice.todense()
    #v,s,u_t=KeCSVD(currentSlice,nHidden)
    #currentVS=v*s
    currentSVD=np.hstack((currentSlice.todense(),currentSVD))
    v,s,u_t=KeCSVD(scipy.sparse.csc_matrix(currentSVD),nHidden)
    
  return (v,s,u_t)


def indirectSVD(M,nHidden):
  #print M.shape
  Q,R=np.linalg.qr(M.todense())
  print M.shape
  #if nHidden>M.shape[0] or nHidden>M.shape[1]:
    
    #nHidden= min(M.shape[0],M.shape[1])-1
  v_r,s,u_t=np.linalg.svd(R)
  v=Q.dot(v_r)

  s=s[0:nHidden]
  v=v[:,0:nHidden]
  u_t=u_t[0:nHidden,:]
  #voglio che s abbia come dimnsione n*m
  #print s
  s=sparse.csc_matrix((s.tolist(),(range(s.shape[0]),range(s.shape[0]))), shape=(v.shape[1],u_t.shape[0]))
  s=s.todense()
  #print "s in indirect SVD",s.shape
  
  
  return (v,s,u_t)

def KeCSVD(M,nHidden=10):
  if M.shape[0]<= M.shape[1]:
    print "ker"
    #eseguo svd su funzione kernel di M
    Kernel=M*M.transpose()
    v,Ssqr,_=indirectSVD(Kernel,nHidden)
    s=np.sqrt(Ssqr)
    u_t=np.linalg.pinv(s)*v.transpose()*M
    #return v,s,u_t
  else:
    print "cov"
    #caso covarianza
    Cov=M.transpose()*M
    _,Ssqr,u_t=indirectSVD(Cov,nHidden)
    s=np.sqrt(Ssqr)
    v=M*u_t.transpose()*np.linalg.pinv(s)
  C=s.shape[0]-1

  for c in reversed(range(C)):
	
        if s[c,c]>0.0001:
	  break
	c=c-1

  c=c+1

  v=v[:,0:c]
  s=s[0:c,0:c]
  u_t =u_t[0:c,:]

  return v,s,u_t

def GetWeightsForHidnOut(sequences, targhet, model,nHidden):
  H=np.empty((0,nHidden), dtype=float)  
  for s in seq:
      (sample,h)=model.predict(s)
      H=np.concatenate((H, h), axis=0)
  H=np.array(H)
  #----ricodifico il targhet come vettore bidimensionale---#
  t=np.empty((0,targhet.shape[2]), dtype=float)
  for song in targhet:
    #print "t",t.shape
    #print "song",song.shape
    t=np.concatenate((t, song), axis=0)
    
  targhet=np.array(t)
  W=np.linalg.pinv(H).dot(targhet)
  return W

  
  
  
#-----Calcolo matrici pesi per autoencoders lineare-----#
def WeightMarixBySVD(vl, sl, u_l,singleSampleDim=88):
    A=(u_l[:,0:singleSampleDim])
    #Definisco R
    #per ogni esempio la prima riga e l'ultima colonna sono a 0
    samplePos=1
    Rrow=list()
    Rcol=list()
    Rdata=list()

    for Clen in sampleLen:
       #per ogni esempio la prima riga e l'ultima colonna sono a 0
       for i in range(samplePos,samplePos+(Clen-1)):
	 Rrow.append(i)
	 Rcol.append(i-1)
	 Rdata.append(1)
       samplePos=samplePos+Clen
    R=sparse.coo_matrix((Rdata,(Rrow, Rcol)), shape=(vl.shape[0],vl.shape[0]))
    Q=sl*(vl.transpose())*R.transpose()*(vl)*np.linalg.inv(sl)
    B=Q.transpose()
    return (A.transpose(), np.asarray(B).transpose(), sl.shape[0])
 

class RNN(object):
    """    Recurrent neural network class

    Supported output types:
    real : linear output units, use mean-squared error
    binary : binary output units, use cross-entropy error
    softmax : single softmax out, use cross-entropy error
    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh,
                 output_type='real', use_symbolic_softmax=False, W_ih=None,  W_hh=None, W_hy=None):

        self.input = input
        self.activation = activation
        self.output_type = output_type

        # when using HF, SoftmaxGrad.grad is not implemented
        # use a symbolic softmax which is slightly slower than T.nnet.softmax
        # See: http://groups.google.com/group/theano-dev/browse_thread/
        # thread/3930bd5a6a67d27a
        if use_symbolic_softmax:
            def symbolic_softmax(x):
                e = T.exp(x)
                return e / T.sum(e, axis=1).dimshuffle(0, 'x')
            self.softmax = symbolic_softmax
        else:
            self.softmax = T.nnet.softmax

        # recurrent weights as a shared variable
        if W_hh==None:
            W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                              low=-.01, high=.01),
                                              dtype=theano.config.floatX)
        else:
            W_init=W_hh
            #W_test = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
            #                                  low=-.01, high=.01),dtype=theano.config.floatX)
            #print W_test.shape

        self.W = theano.shared(value=W_init, name='W')
        # input to hidden layer weights
        if W_ih==None:
             W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        else:
            W_in_init=W_ih

       
        self.W_in = theano.shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        if W_hy==None:
	    W_out_init = np.asarray(np.random.uniform(size=(n_hidden, n_out),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX)
                     
        else:
	    W_out_init=W_hy

          
        self.W_out = theano.shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)

        self.h0 = theano.shared(value=h0_init, name='h0')

        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by')

        self.params = [self.W, self.W_in, self.W_out, self.h0,
                       self.bh, self.by]

        # for every parameter, we maintain it's last update
        # the idea here is to use "momentum"
        # keep moving mostly in the same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # recurrent function (using tanh activation function) and linear output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        [self.h, self.y_pred], _ = theano.scan(step,
                                               sequences=self.input,
                                               outputs_info=[self.h0, None])

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            # push through sigmoid
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)  # apply sigmoid
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            self.p_y_given_x = self.softmax(self.y_pred)

            # compute prediction as class whose probability is maximal
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_out', self.y_out.type))

        if self.output_type in ('binary', 'softmax'):
            # check if y is of the correct datatype
            if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                return T.mean(T.neq(self.y_out, y))
            else:
                raise NotImplementedError()


class MetaRNN(BaseEstimator):
    def __init__(self, n_in=5, n_out=5, nHidden=5, learning_rate=0.01,
                 n_epochs=100, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=1,
                 activation='tanh', output_type='real',
                 final_momentum=0.9, initial_momentum=0.5,
                 momentum_switchover=5,
                 use_symbolic_softmax=False, A=None, B=None, hiddenWeights=None):
        self.n_in = int(n_in)

        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.use_symbolic_softmax = use_symbolic_softmax

	self.ready(hiddenWeights=hiddenWeights)

    def ready(self, hiddenWeights=None):
        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=theano.config.floatX)
        elif self.output_type == 'binary':
            self.y = T.matrix(name='y', dtype='int32')
        elif self.output_type == 'softmax':  # only vector labels supported
            self.y = T.vector(name='y', dtype='int32')
        else:
            raise NotImplementedError
        # initial hidden state of the RNN
        self.h0 = T.vector()
        # learning rate
        self.lr = T.scalar()

	if self.activation == 'lin':
	    activation = lambda x: x
        elif self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError
            
       
       
 



        self.rnn = RNN(input=self.x, n_in=self.n_in,
                       n_hidden=nHidden, n_out=self.n_out,
                       activation=activation, output_type=self.output_type,
                       use_symbolic_softmax=self.use_symbolic_softmax,  W_ih=A,  W_hh=B, W_hy=hiddenWeights)

        if self.output_type == 'real':
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=[self.rnn.y_pred,self.rnn.h],
                                           mode=mode)
        elif self.output_type == 'binary':
            self.predict_proba = theano.function(inputs=[self.x, ],
                                outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                outputs=[T.round(self.rnn.p_y_given_x),self.rnn.h],
                                mode=mode)
        elif self.output_type == 'softmax':
            self.predict_proba = theano.function(inputs=[self.x, ],
                        outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                outputs=self.rnn.y_out, mode=mode)
        else:
            raise NotImplementedError


    def shared_dataset(self, data_xy):
        
        data_x, data_y = data_xy


        shared_x = theano.shared(data_x)

        shared_y = theano.shared(data_y)

        if self.output_type in ('binary', 'softmax'):
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        params = self._get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.rnn.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.rnn.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.

        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validation_frequency=100):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.tanh

        X_train : ndarray (n_seq x n_steps x n_in)
        Y_train : ndarray (n_seq x n_steps x n_out)

        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        """
        if X_test is not None:
            assert(Y_test is not None)
            self.interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            self.interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        n_train = train_set_x.get_value(borrow=True).shape[0]
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[0]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.rnn.loss(self.y) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.rnn.loss(self.y),
                                              givens={
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
            mode=mode)

        if self.interactive:
            compute_test_error = theano.function(inputs=[index, ],
                        outputs=self.rnn.loss(self.y),
                        givens={
                            self.x: test_set_x[index],
                            self.y: test_set_y[index]},
                        mode=mode)

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.rnn.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(self.rnn.params, gparams):
	  
            weight_update = self.rnn.updates[param]
            upd =mom * weight_update - l_r * gparam
            updates[weight_update] = upd
            updates[param] = param + upd
        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[index, l_r, mom],
                                      outputs=[cost,self.rnn.h],
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x[index],
                                          self.y: train_set_y[index]},
                                          mode=mode)

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0
        x1= time.strftime('%s')
        #ftr=open('./RNN_ACC_Train_Pre_TESTDS_JSB150', 'w+')
        #fts=open('./RNN_ACC_Test_Pre_TESTDS_JSB150', 'w+')
        #ftrValid=open('./RNN_ACC_Valid_Pre_TESTDS_JSB150', 'w+')
      
        while (epoch < self.n_epochs):
            epoch = epoch + 1

	    #CICLA SUGLI ESEMPI!!!
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost,h = train_model(idx, self.learning_rate,
                                           effective_momentum)
                #print h.shape
		

                # iteration number (how many weight updates have we made?)
                # epoch is 1-based, index is 0 based
                iter = (epoch - 1) * n_train + idx + 1
                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i)
                                    for i in xrange(n_train)]
                    this_train_loss = np.mean(train_losses)

                    if self.interactive:
                        test_losses = [compute_test_error(i)
                                        for i in xrange(n_test)]
                        this_test_loss = np.mean(test_losses)

                        logger.info('epoch %i, seq %i/%i, tr loss %f '
                                    'te loss %f lr: %f' % \
                        (epoch, idx + 1, n_train,
                         this_train_loss, this_test_loss, self.learning_rate))
                    else:
                        logger.info('epoch %i, seq %i/%i, train loss %f '
                                    'lr: %f' % \
                                    (epoch, idx + 1, n_train, this_train_loss,
                                     self.learning_rate))

            self.learning_rate *= self.learning_rate_decay
	    #if epoch%100==0 or epoch<=100:
	      
		  #accEval(self,str(epoch),ftr,fts,ftrValid)

            

        x2 = time.strftime('%s')
        timediff = int(x2) - int(x1)
        print "Tempo training:", timediff 
        
        
def loadDataSetMin(files):
    #File e il path della carterlla contente i file (*.mid)
    assert len(files) > 0, 'Training set is empty!' \
                           ' (did you download the data files?)'
    #mi calcolo quel'el 'esempio di lunghezza massima
    minLen=sys.maxint
    dataset=[]
    for f in files:
        currentMidi=midiread(f, (21, 109),0.3).piano_roll.astype(theano.config.floatX)
        dataset.append(currentMidi)
        if minLen> currentMidi.shape[0]:
            minLen=currentMidi.shape[0]
    #porto tutte le tracce a masima lunghezza aggiongendo silenzio
    for i, seq in enumerate(dataset):
            if seq.shape[0]>minLen:
                dataset[i]=seq[0:minLen,:] 
                        
    #print dataset[0].shape
    #print "MINLEN: ", minLen
    return np.array(dataset, dtype=theano.config.floatX)

def loadDataSet(files):
    #File e il path della carterlla contente i file (*.mid)
    assert len(files) > 0, 'Training set is empty!' \
                           ' (did you download the data files?)'
    #mi calcolo quel'el 'esempio di lunghezza massima
    maxLen=0
    dataset=[]
    for f in files:
        currentMidi=midiread(f, (21, 109),0.3).piano_roll.astype(theano.config.floatX)
        dataset.append(currentMidi)
        if maxLen< currentMidi.shape[0]:
            maxLen=currentMidi.shape[0]
    #porto tutte le tracce a masima lunghezza aggiongendo silenzio
    for i, seq in enumerate(dataset):
	    print "shape",seq.shape
            if seq.shape[0]<maxLen:
                dataset[i]=np.concatenate([seq, np.zeros((maxLen-seq.shape[0], 88))])
                        
    #print dataset[0].shape
    print "MAXLEN: ",maxLen
    return np.array(dataset, dtype=theano.config.floatX)

def reconstruction(y_T,A,B,T,H):
  #ho y_t moltiplicando per A' ottengo x_t (su quale calcolo l'errore)
  #moltiplicando per B ottengo y_t- dove richiamo il processo
  reco_x=[]
  y_t=np.matrix(y_T)
  
  print y_t.shape
  for i in range(0,T):  
    x_t=y_t*A.transpose()
    x_forSave=(np.matrix.round(np.abs(np.squeeze(np.asarray(x_t)))))
    reco_x.append(x_forSave.tolist())
    y_t= y_t*B.transpose()

  return reco_x

def RecoAccurancy(groundTruth=None, generate=None,InputDim=88):
   true=0
  
 
   assert  len(groundTruth) == len(generate), 'La dimensione dei due input non corrisposnde'
   for noteGT,noteG in zip(groundTruth,generate):
     if noteGT==noteG:
       true=true+1
   return true/InputDim
def slice_sampler(px, N = 1, x = None):
    """
    Provides samples from a user-defined distribution.
    
    slice_sampler(px, N = 1, x = None)
    
    Inputs:
    px = A discrete probability distribution.
    N  = Number of samples to return, default is 1
    x  = Optional list/array of observation values to return, where prob(x) = px.
 
    Outputs:
    If x=None (default) or if len(x) != len(px), it will return an array of integers
    between 0 and len(px)-1. If x is supplied, it will return the
    samples from x according to the distribution px.    
    """
    values = np.zeros(N, dtype=np.int)
    samples = np.arange(len(px))
    px = np.array(px) / (1.*sum(px))
    u = uniform(0, max(px))
    for n in xrange(N):
        included = px>=u
        choice = random.sample(range(np.sum(included)), 1)[0]
        values[n] = samples[included][choice]
        u = uniform(0, px[included][choice])
    if x:
        if len(x) == len(px):
            x=np.array(x)
            values = x[values]
        else:
            print "px and x are different lengths. Returning index locations for px."
    if N == 1:
        return values[0]
    return values

def evaluateAcc (groundTruth=None, generate=None ):
    #Poniamo vero che i parametri in input siano dei piano-roll, quindi abbiamo un vettore che contiene piu vettori di 88 valori (o cmq io considero solo i primi 88 (che sono r[0]-r[1])

    assert  len(groundTruth) == len(generate), 'La dimensione dei due input non corrisposnde'
    maxPoli=0
    clipPoly=[]
    GtPoly=[]
    TP=0.0
    FP=0.0
    FN=0.0
    for frame, gt in zip(groundTruth, generate):
       
        #calcolo massima polifonia della clip
        currentPoliGT=0
        currentPoliGen=0
        for i in xrange(len(frame)):
            if frame[i]==1:
                currentPoliGen=currentPoliGen+1
                if gt[i]==1:
                    TP=TP+1
                    currentPoliGT=currentPoliGT+1
                if gt[i]==0:
                    FP=FP+1
            else:
                if gt[i]==1:
                    currentPoliGT=currentPoliGT+1
#                    else:
#                        #vuol dire che sia frame[i] che gt[i] sono uguali a 0, quindi devo aumentare i TP
#                        TP=TP+1
        clipPoly.append(currentPoliGen)
        GtPoly.append(currentPoliGT)
        if currentPoliGT>maxPoli:
            maxPoli=currentPoliGT
    
    for i, frame in enumerate(groundTruth):
        FN=FN+(maxPoli-clipPoly[i])-(maxPoli-GtPoly[i])
#        print 'TP=', TP
#        print 'FP=', FP
#        print 'FN=', FN
    ACC=0.0
    if TP+FP+FN==0:
      FP=0.0001
    ACC=TP/(TP+FP+FN)
    return ACC
 
 
def accEval(model,epoch,ftr,fts,fvl):
  seq=loadDataSetMin(trainingSet)
 
  targets=[]
  for  s in seq:
    s=np.roll(s,-1,axis=0)
    s[len(s)-1]=np.zeros(s.shape[1])
    targets.append(s)
  targets=np.array(targets)
  
  samples=[]
  for seq in targets:
      sample=model.predict_proba(seq)
      samples.append(sample)
        
  for n, sample in enumerate(samples):
        for j, row in enumerate(sample):
            for k, element in enumerate(row):
                sample[j][k]= slice_sampler(px=[element, 1-element], N=1, x=[1, 0])
    
    
  Acc=[]
  
  for i, seq in enumerate(targets):
      #for i in range(10):
      
      value=evaluateAcc(seq, samples[i])
      Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
  print>>ftr,epoch,' : ',  np.mean(Acc)
  
  #---Load Test Set and evaluating the accuracy----#
  test_set=loadDataSetMin(testSet)
  samples=[]
  
  for seq in test_set:
      sample=model.predict_proba(seq)
      samples.append(sample) 
      
  for n, sample in enumerate(samples):
        for j, row in enumerate(sample):
            for k, element in enumerate(row):
                sample[j][k]= slice_sampler(px=[element, 1-element], N=1, x=[1, 0])
  Acc=[]
  
  for i, seq in enumerate(test_set):
      #for i in range(10):
      value=evaluateAcc(seq, samples[i])
      Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
  print >>fts,epoch,' : ', np.mean(Acc)
  
  #---Load Valid Set and evaluating the accuracy----#
  test_set=loadDataSetMin(validSet)
  samples=[]
  
  for seq in test_set:
      sample=model.predict_proba(seq)
      samples.append(sample) 
      
  for n, sample in enumerate(samples):
        for j, row in enumerate(sample):
            for k, element in enumerate(row):
                sample[j][k]= slice_sampler(px=[element, 1-element], N=1, x=[1, 0])
  Acc=[]
  
  for i, seq in enumerate(test_set):
      #for i in range(10):
      value=evaluateAcc(seq, samples[i])
      Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
  print >>fvl,epoch,' : ', np.mean(Acc)
  
def savehiddenstate(sequences,model,fileName):
  firstLayer=[]
  for sequence in sequences:
     (_,h)=model.predict(sequence)
     firstLayer.append(h)
  np.save(fileName,np.array(firstLayer))
  
  
  
if __name__ == "__main__":
  n_epochs=5000
  InputDim=88
  nHidden=150
  #*--------TEST DATA-------*#
  #dataset=[np.array([[1,0,1,1],[0,1,1,1],[0,0,0,1]]),np.array([[ 1,1,1,1],[1,1,0,1],[0,0,1,1]])]
  #nSample = 2
  ##maxLen=3
  #sampleLen=[3,3]
  #InputDim=4
  #*--------END TEST DATA-------*#
  
  dataset,  sampleLen,  nSample, maxLen = LoadDataForPreTraining()
  #Create autoencoders matrix M
  M=CreateMatrixForSvd(dataset,sampleLen,  nSample, maxLen, InputDim)

  #print M.todense()
  #Calculate SVD(M)
  
  #x1 = time.strftime('%s')
  #(v,s,u_t)=indirectSVD(M)
  #x2 = time.strftime('%s')
  #timediff = int(x2) - int(x1)
  #print 'MySVD:', timediff
  
  #x1 = time.strftime('%s')
  #(v,s,u_t)=np.linalg.svd(M.todense())
  #x2 = time.strftime('%s')
  #timediff = int(x2) - int(x1)
  #print 'np.SVD:', timediff
  
  #print v
  #raw_input("Press enter to exit")
  #print s
  #raw_input("Press enter to exit")
  #print u_t
  xpt1 = time.strftime('%s')
  (v,s,u_t)=SvdForBigData(M,88,nHidden)
  #print "dim after psTroncatedSvd"
  #print v.shape
  #print s.shape
  #print u_t.shape
  #print v
  #print s
  #print u_t
  #Calculate Matrix of aoutencoders Weights
  (A,B, nHidden)=WeightMarixBySVD(v,s,u_t,InputDim)

  #*--------TEST DATA-------*#
  #seq1=np.array([[1,0,1,1],[0,1,1,1],[0,0,0,1]],dtype=theano.config.floatX)#dataset
  #seq2=np.array([[ 1,1,1,1],[1,1,0,1],[0,0,1,1]],dtype=theano.config.floatX)
  #seq=np.array([seq1,seq2],dtype=theano.config.floatX)
  ##*--------END TEST DATA-------*#
  seq=loadDataSetMin(trainingSet)
 
  targets=[]
  for  s in seq:
    s=np.roll(s,-1,axis=0)
    s[len(s)-1]=np.zeros(s.shape[1])
    targets.append(s)
  targets=np.array(targets)
  
  #model = MetaRNN(n_in=InputDim, n_out=InputDim, nHidden=nHidden,
                    #learning_rate=0.001, learning_rate_decay=0.999,
                    #n_epochs=n_epochs, activation='tanh', output_type='binary', A=A, B=B, hiddenWeights=None)
  
  #----Pretraining Hidden To Output Weights----#
  #hiddenWeights=GetWeightsForHidnOut(seq,targets,model,nHidden)
  xpt2 = time.strftime('%s')
  timediff1 = int(xpt2) - int(xpt1)
  
  model = MetaRNN(n_in=InputDim, n_out=InputDim, nHidden=nHidden,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=n_epochs, activation='tanh', output_type='binary', A=A, B=B, hiddenWeights=None)
      

  #TRAINING
  xt1 = time.strftime('%s')
  model.fit(seq, targets, validation_frequency=1000)
  xt2 = time.strftime('%s')
  print "save output"
  savehiddenstate(seq,model,"TrainingSet.txt")
  print "save output Test"
  test_set=loadDataSetMin(testSet)
  savehiddenstate(test_set,model,"TestSet.txt")
  print "save output Valid"
  valid_set=loadDataSetMin(validSet)
  savehiddenstate(valid_set,model,"ValidSet.txt")
  
  timediff = int(xt2) - int(xt1)
  
  #ftime=open('./TimeWithPreT', 'w+')
  #print>>ftime, 'Traning: ',  timediff, 'Pre-Traning: ',  timediff1
  #Calcolo il valore delle hidden units dell'autoencoders
  #samples=[]
  ##H=[]
  ##for s in dataset:
      ##(sample,h)=model.predict(s)
      ##samples.append(sample)
      ##H.append(h)
  
  #samples=[]
  #for seq in targets:
      #sample=model.predict_proba(seq)
      #samples.append(sample)
        
  #for n, sample in enumerate(samples):
        #for j, row in enumerate(sample):
            #for k, element in enumerate(row):
                #sample[j][k]= slice_sampler(px=[element, 1-element], N=1, x=[1, 0])
    
    
  #Acc=[]
  #f=open('./RNN_ACC_Train_Pre_AccTest_JSB100', 'w+')
  #for i, seq in enumerate(targets):
      ##for i in range(10):
      
      #value=evaluateAcc(seq, samples[i])
      #Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
  #print>>f, 'Acc on Train:',  np.mean(Acc)
  
  ##---Load Test Set and evaluating the accuracy----#
  #test_set=loadDataSetMin(testSet)
  #samples=[]
  
  #for seq in test_set:
      #sample=model.predict_proba(seq)
      #samples.append(sample) 
      
  #for n, sample in enumerate(samples):
        #for j, row in enumerate(sample):
            #for k, element in enumerate(row):
                #sample[j][k]= slice_sampler(px=[element, 1-element], N=1, x=[1, 0])
  #Acc=[]
  #f=open('./RNN_ACC_Test_Pre_AccTest_JSB100', 'w+')
  ##seq=seq[1:]
  #for i, seq in enumerate(test_set):
      ##for i in range(10):
      #value=evaluateAcc(seq, samples[i])
      #Acc.append(value)
      #print >>f, ('esempio:', i, 'Acc:', value)
  #print >>f,'Acc on test:', np.mean(Acc)
  
  
  
  #Valid
  #eseguo la ricostruzione
  #for i,h in enumerate(H):
      ##Y_T e il valore delle Hidden all'ultimo istante di tempo
      #y_T= h[h.shape[0]-1,:]
      #x_reco=reconstruction(y_T, A, B, len(seq[i]),h)  
      #x_reco.reverse()
      #res=1
      #for j,gt in enumerate(seq[i].tolist()):
	##print x_reco[j]
	#res=res*RecoAccurancy(gt,x_reco[j],InputDim)
	#print "result:",res



  
  
