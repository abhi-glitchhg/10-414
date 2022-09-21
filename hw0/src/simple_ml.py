import struct
import numpy as np
import gzip
from numpy.lib import diagflat

from numpy.lib.npyio import DataSource
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x+y
    ### END YOUR CODE


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    img_file = gzip.open(image_filesname)
    img_file.read(4) #no need to read magic number,
    num_images_buf = img_file.read(4)
    num_images = struct.unpack_from(">I",num_images_buf)[0]
    img_file.read(8)

    buffer_ = img_file.read(num_images * 784) #60000 * 28*28 data read  
    img_file.close()
    data = np.frombuffer(buffer_,dtype=np.uint8).astype(np.float32)
    imgs = data.reshape(num_images, 784)
    imgs = imgs/255

    """
    #another way to do would be like this, but above method is faster.
    buffer_ = img_file.read(47040000)
    data = []
    for i in range(47040000):
      data.append(struct.unpack_from(">I",buffer_)[0])
    data = np.array(data).astype(np.float32)
    imgs=data.reshape(60000, 784)
    del data"""

    label_file = gzip.open(label_filename)
    label_file.read(8)

    buffer_ = label_file.read(num_images)
    label_file.close()
    labels = np.frombuffer(buffer_,dtype=np.uint8)
    #print(labels.dtype)
    return imgs, labels

    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
 
    n_classes= Z.shape[1]
    mask = np.eye(n_classes)[y]

    return np.mean(np.log(np.sum(np.exp(Z),axis=1)) - Z[mask==1.])
    
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameter, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n_classes = theta.shape[1]
    num_examples = X.shape[0]

    for i in range(0,num_examples,batch):
      #fetch the values for the iteration
      x = X[i:i+batch] 
      yi= y[i:i+batch]
      
      #matrix multiplication 
      prod  = np.exp( x@theta)
      #normalisation
      z = prod/ (prod@np.ones((n_classes,1)))
      
      # Iy matrix trick 
      i = np.eye(batch, n_classes)[yi]
      
      #calculate grad wrt theta
      grad = (x.T @ (z-i))/ batch
      
      #update theta.
      theta-= lr* grad
    
    

    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarrray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarrray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    ### BEGIN YOUR CODE
    
    n_classes = W2.shape[1]
    num_examples= X.shape[0]
    hidden_dim = W1.shape[1]
    relu = lambda t: np.maximum(0,t)
    signum = lambda t: (np.sign(t)+1)//2 #this signum is basically differentiation of relu, if x>0 returns 1 else returns 0.
    #print(f"batch, {batch}")
    for i in range(0,num_examples, batch):
      x = X[i:i+batch]
      yi =y[i:i+batch]

      out1 = x @ W1 
      act1 = relu(out1)
      out2 = act1 @ W2

      S =  np.exp(out2)/ (np.exp(out2)@np.ones((n_classes,1)))

      I =  np.eye(batch, n_classes)[yi]
      grad2 = act1.T @ (S-I)/batch
      grad1 = x.T @ ((signum(out1)*((S-I)@W2.T)))/batch

     
      W1 -= lr * grad1
      W2 -= lr * grad2




      pass

    ### END YOUR CODE

def relu(x):
  if x>=0:
    return x 
  return 0

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
