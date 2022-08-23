#1. Closed Form Solution of Linear Regression

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
def closed_form(X, Y, lambda_factor):

  # newX=X^T

  newX=np.transpose(X)

  # xterm = (X^T.X + lambda_factor*I)

  xterm=np.dot(newX,X)+(lambda_factor*np.identity(newX.shape[0]))

  # xterm = xterm's inverse

  xterm=np.linalg.inv(xterm)

  # yterm = X^T.Y

  yterm=np.dot(newX,Y)

  # returning xterm.yterm = (X^T.X + lambda_factor*I).(X^T.Y)

  return np.dot(xterm,yterm)
  
  #2. One vs. Rest SVM
  
  def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(C = 0.1, random_state = 0)
    clf.fit(train_x, train_y)
    
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y
    
    # 3.Multiclass SVM

def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(C = 0.1, random_state = 0)
    clf.fit(train_x, train_y)
    
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y 
#pragma: coderesponse end


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - (pred_test_y == test_y).mean()
    
    #4. Multinomial (Softmax) Regression and Gradient Descent
    
def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    # Compute the matrix of theta*X' (each row is a category, column an example)
    R = (theta.dot(X.T))/temp_parameter
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    
    # Compute H matrix
    H = np.exp(R - c)
    
    # Divide H by the normalizing term
    H = H/np.sum(H, axis = 0)
    
    return H    
#pragma: coderesponse end

# Cost Function

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # avg error term
    
    # Clip prob matrix to avoid NaN instances
    clip_prob_matrix = np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15)
    
    # Take the log of the matrix of probabilities
    log_clip_matrix = np.log(clip_prob_matrix)
    
    # Create a sparse matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()
    
    # Only add terms of log(matrix of prob) where M == 1
    error_term = (-1/n)*np.sum(log_clip_matrix[M == 1])    
                
    # Regularization term
    reg_term = (lambda_factor/2)*np.linalg.norm(theta)**2
    
    return error_term + reg_term
    
#pragma: coderesponse end


# Gradient Descent

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # Create spare matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    
    # Matrix of Probabilities
    P = compute_probabilities(X, theta, temp_parameter)
    
    # Gradient matrix of theta
    grad_theta = (-1/(temp_parameter*n))*((M - P) @ X) + lambda_factor*theta
    
    # Gradient descent update of theta matrix
    theta = theta - alpha*grad_theta
    
    return theta
    
    #6. Changing Labels
    
    #Using the Current Model - update target

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    return np.mod(train_y, 3), np.mod(test_y, 3)

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    pred_Y = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(np.mod(pred_Y, 3) == Y)

# 8. Dimensionality Reduction Using PCA

def project_onto_PC(X, pcs, n_components, feature_means):
    """
    Given principal component vectors pcs = principal_components(X)
    this function returns a new data array in which each sample in X
    has been projected onto the first n_components principcal components.
    """
    # TODO: first center data using the feature_means
    # TODO: Return the projection of the centered dataset
    #       on the first n_components principal components.
    #       This should be an array with dimensions: n x n_components.
    # Hint: these principal components = first n_components columns
    #       of the eigenvectors returned by principal_components().
    #       Note that each eigenvector is already be a unit-vector,
    #       so the projection may be done using matrix multiplication.
    centered_X = center_data(X)
    return centered_X @ pcs[:,:n_components]
    
    #10. Kernel Methods 
    def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return (X @ Y.T + c)**p

#11. Activation Functions

#Rectified Linear Unit
def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0, x)
    
#Taking the Derivative
def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x <= 0:
        return 0
    else:
        return 1
        


#12. Training the Network

#Implementing Train

class NeuralNetwork(NeuralNetworkBase):

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights.dot(input_values) + self.biases #(3 by 1 matrix)
        ReLU_vec = np.vectorize(rectified_linear_unit)  # Vectorize ReLU function
        hidden_layer_activation = ReLU_vec(hidden_layer_weighted_input) #(3 by 1 matrix)

        output = self.hidden_to_output_weights.dot(hidden_layer_activation)
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = -(y - activated_output) #dC/df(u1)
        
        output_derivative_vec = np.vectorize(output_layer_activation_derivative)    # Vectorize derivative of output activation
        hidden_layer_error = np.multiply(output_derivative_vec(activated_output),self.hidden_to_output_weights.transpose())*output_layer_error #(3 by 1 matrix)
        
        ReLU_derivative_vec = np.vectorize(rectified_linear_unit_derivative) # Vectorize ReLU derivative
        bias_gradients = np.multiply(hidden_layer_error, ReLU_derivative_vec(hidden_layer_weighted_input)) #dC/db

        hidden_to_output_weight_gradients = np.multiply(hidden_layer_activation, output_layer_error).transpose() #dC/dV
        input_to_hidden_weight_gradients = bias_gradients.dot(input_values.transpose()) #dC/dW

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate*bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients

#13. Predicting the Test Data
#Implementing Predict

class NeuralNetwork(NeuralNetworkBase):

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = self.input_to_hidden_weights.dot(input_values) + self.biases
        ReLU_vec = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = ReLU_vec(hidden_layer_weighted_input)
        output = self.hidden_to_output_weights.dot(hidden_layer_activation)
        activated_output = output_layer_activation(output)

        return activated_output.item()

#14. Convolutional Neural Networks

model = nn.Sequential(
nn.Conv2d(1, 32, (3, 3)),
nn.ReLU(),
nn.MaxPool2d((2, 2)),
nn.Conv2d(32, 64, (3, 3)),
nn.ReLU(),
nn.MaxPool2d((2, 2)),
Flatten(),
nn.Linear(1600, 128),
nn.Dropout(0.5),
nn.Linear(128, 10),
)

#15. Overlapping, multi-digit MNIST
#Fully connected network

class MLP(nn.Module):

    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 20)
        
    def forward(self, x):
        xf = self.flatten(x)
        xl1 = self.linear1(xf)
        xl2 = self.linear2(xl1)
        out_first_digit = xl2[:,:10]
        out_second_digit = xl2[:,10:]
        
        return out_first_digit, out_second_digit

#Convolutional model
class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, (3, 3))
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d((2,2))
        self.conv2d_2 = nn.Conv2d(32, 64, (3, 3))
        self.flatten = Flatten() 
        self.linear1 = nn.Linear(2880, 64)
        self.dropout = nn.Dropout(p = 0.5)
        self.linear2 = nn.Linear(64, 20)
        
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.maxpool2d(x) 
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        out_first_digit = x[:,:10]
        out_second_digit = x[:,10:]
        
        return out_first_digit, out_second_digit
