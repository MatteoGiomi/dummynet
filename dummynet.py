# DIY implementation of a fully connected neural network
# math studied here: http://neuralnetworksanddeeplearning.com/

import time, random, tqdm
import numpy as np
np.random.seed(42)


# micro optimize 
exp = np.exp
dot = np.dot
asarray = np.asarray
maximum = np.maximum

# array casting in right shape
def colvec(x):
    """
        turn an input array-like vector into a 1D colum vector
    """
    x = asarray(x)
    x.shape = (len(x), 1)
    return x


# activation functions
def sigmoid(x):
    return ( 1 / ( 1 + exp(-x)) )
def sigmoid_der(x):
    sx = sigmoid(x)
    return sx*(1 - sx)

def relu(x):
    return maximum(0.1*x, x)
#    return maximum(0.1*x, x)
def relu_der(x):
    return 0.1*(x<0) + 1*(x>0)
#    return (x>0).astype(np.int)

class dummylayer:
    
    dtype = np.float32
    
    def __init__(self, n, n_prev, num, activation=None):
        """
            Initialize dummynet layer with random weights. 
            A layer with n inputs and m nodes can be seen as a function f:
                
                f(x, W, b): R^n --> R^m
            
            Where: 
                x is an input vector of length n
                W is an n x m matrix 
                b is a column vector of length m
        """
        self.id     = num
        self.n      = n
        self.w      = np.random.normal(size=(n, n_prev)).astype(self.__class__.dtype)
        self.b      = np.random.normal(size=(n, 1)).astype(self.__class__.dtype)
        
        # pick activation function and its derivative
        if activation in ['sigmoid', None]:
            self.activation = sigmoid
            self.activation_der = sigmoid_der
        elif activation.lower() == 'relu':
            self.activation = relu
            self.activation_der = relu_der

    def __call__(self, x):
        """
            Compute output vector given input x. 
        """
        # make shure the input is interpreted as one-col matrix# TODO: measure inpact on performances
        self.x = colvec(x)
        self.zin = dot(self.w, x) + self.b
        self.out = self.activation(self.zin)
        return self.out

    def compute_gradients(self, err_next, weights_next):
        """
            Computes the derivatives of the cost function C with respect to the 
            parameters (weights W and biases b) of this layer. 
            
            
            Use backpropagation to compute these quantities from the 'error' vector 
            of the next (successive, closer to output) layer. The error is defined 
            as the partial derivative of cost function with respect to the weighted 
            input (Wx+b) of the layer.
            
            Parameters:
            -----------
                
                err_next: `array-like`
                    error vector for next layer.
                
                weight_next: `matrix-like`
                    weight matrix of next layer.
                
                learning_rate: `float`
                    learning rate for the training phase.
            
            Returns:
            --------
                
                error, weights: `array-like` and `matrix-like` 
                    error vector and original weights (BEFORE UPDATE) for this layer.
        """
        error = dot(weights_next.T, err_next) * self.activation_der(self.zin)
        self.set_gradients(error)
        return (error, self.w)
    
    def set_gradients(self, error):
        """
            Compute derivative of cost function wrt weights and biases of this layer
            starting from the error vector. Save the results as attributes.
        """
        self.db = error
        self.dw = dot(error, self.x.T)


class dummynet:
    """
        DIY implementation of a fully connected neural network.
    """
    
    def __init__(self, structure, activation):
        """
            Initialize the dummynet with the chosen architecture.
            
            Parameters:
            -----------
                
                structure: `array-like`
                    1D array specifiying the number of nodes in each of the 
                    layers of the network.
                
                activation: `str` or `array-like`
                    activation function of each layer in the net. If `str` all
                    nodes will have the same activations, pass an 'array-like' 
                    object to specify a different activation for each layer.
            
            Example:
            --------
                
                net = dummynet(2, [3, 4, 3], 1, 'reLU')
                
                creates a network with 2 input and 1 output nodes with 3 hidden layers
                with 3, 4, and 3 nodes each and uses reLU as the activation function
                of all the nodes. 
        """
        
        print("Initializing dummynet.")
        
        if type(activation) == str:
            activation = [activation]*len(structure)
        
        self.structure = structure
        self.layers = []
        for il in range(0, len(structure)):
            act = activation[il]
            if il == 0:
                new_lyr = dummylayer(structure[il], structure[il], num=il, activation=act)
            else:
                new_lyr = dummylayer(structure[il], structure[il-1], num=il, activation=act)
            self.layers.append(new_lyr)


    def __call__(self, x):
        """
            Compute network response to input vector x.
            
            Parameters:
            -----------
                
                x: `array-like`
                    input vector for the network. Will be casted to 1D column vector.
            
            Returns:
            --------
                np.array, 1D with length equal to the number of nodes in the last layer.
        """
        aux = colvec(x)    # cast to column vector
        for layer in self.layers:
            aux = layer(aux)
        return aux


    def cost(self, x, target):
        """
            Evaluate summed squared error cost function for given input and target
            
            Parameters:
            -----------
                
                x: `array-like`
                    input vector
                
                target: `array-like`
                    target array. The cost function will evaluate the deviations
                    of the network output with respect to this vector.
            
            Returns:
            --------
                
                summed squared error of network output wrt target.
        """
        return 0.5*np.sum((self(x)-target)**2)


    def backprop(self, x, target):
        """
            compute error on last layer and propagate it backwards.
        """
        error = (
            (self.layers[-1].out - target) * self.layers[-1].activation_der(self.layers[-1].zin))
        weights = self.layers[-1].w
        self.layers[-1].set_gradients(error)
        for layer in reversed(self.layers[:-1]):
            error, weights = layer.compute_gradients(error, weights)


    def train(self, inputs, targets, learning_rate, n_batches, batch_size):
        """
            Train the dummynet on a given set of labeled inputs using SGD algorithm.
            
            The training sample (inputs, targets) is split into randomly chosen bathces
            of given size. 
            
            For each batch, we first compute, for each training sample, the gradients 
            of the cost function with respect to all the parameters in the network.
            These gradients are then averaged together and used to update the weigths
            and biases of each layer.
            
            Update parameters of this layer applying gradient descent equations:
            
                W --> W - r*SUM(dC/dW)/batch_size
                b --> b - r*SUM(dC/db)/batch_size
            
            where r is the learning rate. 
            
            Parameters:
            -----------
            
                inputs: `array-like`
                    each element of this vector is an array of length matching the 
                    number of nodes in the first layer of the network.
                
                targets: `array-like`
                    desired output of the network for each of the input vectors. The
                    length of each element in targets has to be equal to the number 
                    of output nodes of the network.
                
                learning_rate: `float`
                    at each iteration the gradients will be scaled by this quantity.
                
                n_batches: `int`
                    number of batches.
                    
                batch_size: `int`
                    number of training samples in each batch.
            
            Returns:
            --------
                
                list with values of cost function at each iteration.
            
        """
        
        if len(inputs) != len(targets):
            return ValueError("Input and target vectors should have same length.")
        
        start = time.time()
        print ("training dummynet using SGD with %d batches of %d elements each."%(n_batches, batch_size))
        
        # convert everything to numpy arrays
        inputs = [colvec(x).astype(dummylayer.dtype) for x in inputs]
        targets = [asarray(t, dtype=dummylayer.dtype) for t in targets]
        
        # loop over the mini batches, save total cost for each batch
        costs = []
        training_set = list(zip(inputs, targets))
        for ib in tqdm.tqdm(range(n_batches)):
            
            # allocate space for average of parameters derivatives
            sum_dw, sum_db = [], []
            for lyr in self.layers:
                sum_dw.append(np.zeros(lyr.w.shape, dtype=dummylayer.dtype))
                sum_db.append(np.zeros(lyr.b.shape, dtype=dummylayer.dtype))
            
            # select a random mini-batch TODO: use np random
            batch = random.sample(training_set, batch_size)
            inputs_batch, targets_batch = zip(*batch)
            
            # now derive gradients for each input/label pair, derive cost function 
            # wrt all the weigths in the net and sum together parameter gradients
            cost_buff = np.zeros(len(inputs_batch))
            for it in range(len(inputs_batch)):
                
                # forward pass (also compute cost and save it) and backward
                cost_buff[it] = self.cost(inputs_batch[it], targets_batch[it])
                self.backprop(inputs_batch[it], targets_batch[it])
                
                # sum up the gradients
                for il, lyr in enumerate(self.layers):
                    sum_dw[il] += lyr.dw
                    sum_db[il] += lyr.db
            
            # update parameters using average from batch
            for il, lyr in enumerate(self.layers):
                lyr.w = lyr.w - learning_rate*sum_dw[il]/batch_size
                lyr.b = lyr.b - learning_rate*sum_db[il]/batch_size
            
            # compute total cost for this batch
            costs.append(np.sum(cost_buff))
        end = time.time()
        print ("dummynet has been trained! took %.2e sec"%(end-start))
        return costs

    def visualize(self):
        """
            make pretty plot to visualize the network structure
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        # draw a circle for each neuron with proportional to bias
        nodes = []
        for l_id, layer in enumerate(self.layers):
            for n_id in range(layer.n):
                xx, yy = l_id-len(self.layers)/2., n_id-layer.n/2.
                rad = 5*layer.b[n_id] + 3
                ax.plot(xx, yy, 'ko', markersize=rad, markerfacecolor='w', markeredgewidth=1.5)
                nodes.append([xx, yy])
        
        # create lines connecting all nodes with one another TODO: linewidth = weights
        for i1 in range(len(nodes)):
            
            for i2 in range(len(nodes)):
                # do not connect nodes on the same layer or in distant layers
                dist = abs(nodes[i1][0]-nodes[i2][0]) 
                if dist>1 or dist == 0:
                    continue
                xx, yy = (nodes[i1][0], nodes[i2][0]), (nodes[i1][1], nodes[i2][1])
                ax.add_line(plt.Line2D(xx, yy, zorder=-666, linewidth=0.05, c='k'))
        
        # center the plot
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()

