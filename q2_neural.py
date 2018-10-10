import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

"""def grad_softmax(z,labels):
    grad = np.zeros(z.shape)
    z = np.multiply(z,labels)
    soft = softmax(z)
    print(z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            for k in range(z.shape[1]):
                if(j==k):
                    grad[i][k] += z[i][j]*soft[i][j]*(1-soft[i][j])
                else:
                    grad[i][k] += -z[i][k]*soft[i][k]*(soft[i][j])
    grad = grad[labels==1]
    return grad"""
    

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    N, _ = data.shape
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    cost = -np.sum(np.log(a2[labels==1]))/N

    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    #raise NotImplementedError
    lambda3 = (a2-labels)/N
    gradW2 = np.dot(a1.T, lambda3)
    #gradb2(np.zeros)
    gradb2 = np.sum(lambda3, axis=0, keepdims=True)
    lambda2 = np.dot(lambda3, W2.T) * sigmoid_grad(a1) 
    gradW1 = np.dot(data.T, lambda2)
    gradb1 = np.sum(lambda2, axis=0, keepdims=True)
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
   
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    #sanity_check()
    #your_sanity_checks()
    pass