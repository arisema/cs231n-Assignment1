from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    N, D = X.shape  # num of taining samples and number of dimentions
    C = W.shape[1] # num classes
    scores = np.zeros((N, C))
    scores = np.dot(X, W)

    for n in range(N):
      scores[n] -= np.max(scores[n]) # numeric stability : http://cs231n.github.io/linear-classify/
      
      softmaxFunction = np.exp(scores[n])/np.sum(np.exp(scores[n])) # perform softmax function for the scores of each example
    
      crossEntropyLoss = -np.log(softmaxFunction[y[n]]) 

      for c in range(C):
        dW[:, c] += X[n] * softmaxFunction[c]
      dW[:,y[n]] -= X[n] # neutralize correct class score
    
    dataLoss = np.mean(crossEntropyLoss)
    regLoss = reg * np.sum(np.square(W))
    loss = dataLoss + regLoss
    dW /= N
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #cross-entropy loss
    N, D = X.shape
    C = W.shape[1]
    scores = np.dot(X, W)
    
    scores -= np.max(scores, axis=1, keepdims=True) # numeric stability : http://cs231n.github.io/linear-classify/
    
    softmaxFunction = np.exp(scores) / np.sum(np.exp(scores), axis = 1, keepdims = True)
    # cost J = 1/m(L(yhat, y))
    crossEntropyLoss = -np.log(softmaxFunction[np.arange(N), y])    
    dataLoss = np.sum(crossEntropyLoss)
    regLoss = reg * np.sum(np.square(W))
    loss = dataLoss + regLoss
    loss /= N
    # back prop: dzl = yhat-y
    softmaxFunction[np.arange(N), y] -= 1 #max prob being 1
    dW = np.dot(X.T, softmaxFunction)
    dW /= N
    dW += dW/N + (reg * 2 * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW