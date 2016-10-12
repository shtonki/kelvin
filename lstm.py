import copy, numpy as np
import matplotlib.pyplot as plt
from gen import gendata

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

DATALEN = 10;

fuck, my, shit = gendata(0, DATALEN);

# training dataset generation
int2binary = {}
sequenceSize = DATALEN - 3;

errSum = 0;

largest_number = pow(2,sequenceSize)
binary = np.unpackbits(
    np.array([list(range(largest_number))],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
alpha = 0.8
input_dim = len(my[0]);
hidden_dim = 16
output_dim = len(shit[0]);

plotme = [],[],[]
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

errorarray = []

# training logic
for j in range(1200):
    '''
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)
    '''
    opn, sX, sy = gendata(j, DATALEN);

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(sequenceSize):
        
        # generate input and output
        X = np.array([
            sX[sequenceSize]
            ]);

        y = np.array([
            sy[sequenceSize]
            ])

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        #d[sequenceSize - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    if (j < 1001):

        future_layer_1_delta = np.zeros(hidden_dim)
        
        for position in range(sequenceSize):
            
            X = np.array([
                sX[sequenceSize]
                ]);
            layer_1 = layer_1_values[-position-1]
            prev_layer_1 = layer_1_values[-position-2]
            
            # error at output layer
            layer_2_delta = layer_2_deltas[-position-1]
            # error at hidden layer
            layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

            # let's update all our weights so we can try again
            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)
            
            future_layer_1_delta = layer_1_delta
        

        synapse_0 += synapse_0_update * alpha
        synapse_1 += synapse_1_update * alpha
        synapse_h += synapse_h_update * alpha    

        synapse_0_update *= 0
        synapse_1_update *= 0
        synapse_h_update *= 0
    
    # print out progress
    if(j > 1000 and j % 1 == 0):
        pClose = layer_2[0][0] * opn * 2;
        aClose = y[0][0] * opn * 2;
        pUpDown = layer_2[0][1];
        aUpDown = y[0][1];

        plotme[0].append(pClose);
        plotme[1].append(aClose);
        plotme[2].append(pClose - aClose);
        errSum += abs(pClose - aClose);
        #print(pUpDown, aUpDown);
        #errorarray.append(dff);

print(errSum/len(plotme[0]));
plt.plot(plotme[0]);
plt.plot(plotme[1]);
#plt.plot(plotme[2]);
#plt.plot([0]*len(plotme[0]));
plt.show();