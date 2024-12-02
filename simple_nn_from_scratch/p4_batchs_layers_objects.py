import numpy as np

inputs = [[1, 2, 3, 4],
          [2, 5, -1, 2],
          [-1.5, 2.7, 3.3, -0.8]]

# layer 1 " single neuron"
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
"layer 1 output dim 3x3"

# layer 2
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]
"layer 2 o/p dim 3x3"

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# input shape is broadcasted by duplicated the row 
print(layer2_outputs)
