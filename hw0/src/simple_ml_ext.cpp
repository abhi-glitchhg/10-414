#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matrix_multiplication(const float* x, const float* theta, float* z , int n,int k, int m){    
      for (int i = 0; i < m; i++) {  // matrix multiplication of m X k with k* n
        for (int g = 0; g < k; g++) {
            for (int j = 0; j < n; j++) {
                z[i * n + j] += x[i * k + g] * theta[g * n + j];
            }
        }
    }}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)

         ### BEGIN YOUR CODE
      
     */

    /// BEGIN YOUR CODE

    //since we dont have data in matrix form; we will have to write the matrix multiplication manually;
    //also allocate memory from heap to store output 

    //z = np.exp(x@theta)

    auto  z= new float[k*batch] ;// z has shape of [batch, k]; hence this much mem;
    auto grad = new float [n*k]; // same size as theta
    //auto  Iy= new float[k*batch];
    auto x_transpose = new float[batch * n];


    for (int I=0; I <m; I+=batch) { //accessing the images with the for loop
    auto x = X + I * n; //X is a pointer to float
    auto yi = y + I; //

    // initialise the vals : saw this in a someone else's solution . Very help full
    std::fill(z, z + batch * k, 0.0f);
    std::fill(x_transpose, x_transpose + batch * n, 0.0f);
    std::fill(grad, grad + n * k, 0.0f);

    matrix_multiplication(x, theta, z,k,n, batch); // z contains the matrix multiplication product

    //now we have to take exponentials  n normalise them
    for (int i=0; i< batch; i++){
          float summ=0.0f;

      for (int j=0; j<k;j++){
        z[i*k+j] = exp(z[i*k+j]);
        summ+= z[i*k + j];
      }

      for (int j=0;j<k;j++)
        z[i*k+j] /=summ;

      
    }
    //normalisation done :)
    //Iy matrix :( (i dearly miss numpy)
    /*
    for (int i=0; i< batch; i++){
      //
      Iy[i*k + y[i]] =1.0f ; //hopefully this works :(
    }

    for (int i=0; i<batch;i++){
      z[i*k +yi[i]] -= Iy[i*k+yi[i]]; 
    }
    i realised we can just subtract one from the z; no need of iy hehe
    */

    // Z - Iy done 
    for (int i=0; i<batch; i++){  
      z[i * k + yi[i]] -= 1.0f;
    }

    //store transpose of x in xT
    for (int i = 0; i < n; i++) {
            for (int j = 0; j < batch; j++) {
                x_transpose[i * batch + j] = x[j * n + i];
            }
        }
    
    matrix_multiplication(x_transpose, z, grad, k, batch, n);

    for (int i=0; i<n*k;i++){
      theta[i] -= lr * grad[i]/batch;
    }

    }    
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}