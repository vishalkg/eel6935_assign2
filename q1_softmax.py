import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """

    ### YOUR CODE HERE
    if x.ndim==1:
        softmax_val = np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
    else:
        softmax_val = np.zeros(x.shape,dtype=np.float128)
        for i in range(0,softmax_val.shape[0]):
            softmax_val[i] = np.exp(x[i]-np.max(x[i]))/np.sum(np.exp(x[i]-np.max(x[i])))
    ### END YOUR CODE
    
    return softmax_val

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6
    
    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6
    
    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6
    
    print ("You should verify these results!\n")

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print ("Running your tests...")

    ### YOUR CODE HERE
    ### Since the values are nothing but probabilities, they should sum to 1
    test1 = softmax(np.array([1,2]))
    assert np.sum(test1) == 1.0

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    assert np.sum(test2)==test2.shape[0]

    test3 = softmax(np.array([[-1001,-1002]]))
    assert np.sum(test2)==test2.shape[0]
    ### END YOUR CODE

    print("All Tests passed !\n")  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
