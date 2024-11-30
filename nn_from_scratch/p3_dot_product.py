######### dynamic for the previous p2 ###########
inputs = [1, 2, 3, 4]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

"""
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, bias):
    neuron_output = 0
    for n_input, weight in zip(input, neuron_weights):
        neuron_output += n_input*weight

    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

"""
import numpy as np

outputs = np.dot(weights, inputs) + bias
# input shape is broadcasted by duplicated the row 
print(outputs)
