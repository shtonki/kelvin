import copy, numpy as np
from gen import gendata


def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output*(1-output)

class lstm:



    def __init__(self, inputs, outputs, learningRate):
        self.alpha = learningRate;
        self.input_dim = inputs;
        self.hidden_dim = 16
        self.output_dim = outputs;

        self.synapse_0 = 2*np.random.random((self.input_dim,  self.hidden_dim)) - 1
        self.synapse_1 = 2*np.random.random((self.hidden_dim, self.output_dim)) - 1
        self.synapse_h = 2*np.random.random((self.hidden_dim, self.hidden_dim)) - 1

        self.synapse_0_update = np.zeros_like(self.synapse_0)
        self.synapse_1_update = np.zeros_like(self.synapse_1)
        self.synapse_h_update = np.zeros_like(self.synapse_h)

    def train(self, sX, sy):
        sequenceSize = len(sX);
        
        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hidden_dim))
        
        for position in range(sequenceSize):
            X = np.array([
                sX[position]
                ]);

            y = np.array([
                sy[position]
                ])

            layer_1 = sigmoid(np.dot(X,self.synapse_0) + np.dot(layer_1_values[-1],self.synapse_h))
            layer_2 = sigmoid(np.dot(layer_1,self.synapse_1))
            layer_2_error = y - layer_2
            layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
            layer_1_values.append(copy.deepcopy(layer_1))

        synapse_0_update = 0
        synapse_1_update = 0
        synapse_h_update = 0
        future_layer_1_delta = np.zeros(self.hidden_dim)
            
        for position in range(sequenceSize):
            
            X = np.array([
                sX[position]
                ]);
            layer_1 = layer_1_values[-position-1]
            prev_layer_1 = layer_1_values[-position-2]
            
            layer_2_delta = layer_2_deltas[-position-1]
            layer_1_delta = (future_layer_1_delta.dot(self.synapse_h.T) + layer_2_delta.dot(self.synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

            synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
            synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
            synapse_0_update += X.T.dot(layer_1_delta)
            
            future_layer_1_delta = layer_1_delta
            

        self.synapse_0 += synapse_0_update * self.alpha
        self.synapse_1 += synapse_1_update * self.alpha
        self.synapse_h += synapse_h_update * self.alpha    

        return layer_2;

        


    def predict(self, sX):
        sequenceSize = len(sX);
        overallError = 0
        
        layer_2_deltas = list()
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.hidden_dim))
        
        for position in range(sequenceSize):
            X = np.array([
                sX[position]
                ]);

            layer_1 = sigmoid(np.dot(X,self.synapse_0) + np.dot(layer_1_values[-1],self.synapse_h))

            layer_2 = sigmoid(np.dot(layer_1,self.synapse_1))

            layer_1_values.append(copy.deepcopy(layer_1))

        return layer_2;

    '''
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
    '''