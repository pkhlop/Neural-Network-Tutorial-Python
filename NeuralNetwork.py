#
# Imports
#
import numpy as np
import sys

#
# Transfer functions
#
import os


class TransferFunctions:
    def sgm(x, Derivative=False):
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            out = sgm(x)
            return out * (1.0 - out)

    def linear(self, x, Derivative=False):
        if not Derivative:
            return x
        else:
            return 1.0

    def gaussian(x, Derivative=False):
        if not Derivative:
            return np.exp(-x ** 2)
        else:
            return -2 * x * np.exp(-x ** 2)

    def tanh(self, x, Derivative=False):
        if not Derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x) ** 2

    def truncLinear(x, Derivative=False):
        if not Derivative:
            y = x.copy()
            y[y < 0] = 0
            return y
        else:
            return 1.0


#
# Classes
#
class BackPropagationNetwork:
    """A back-propagation network"""

    #
    # Class methods
    #
    def __init__(self, layerSize, layerFunctions=None):
        """Initialize the network"""

        self.layerCount = 0
        self.shape = None
        self.weights = []
        self.tFuncs = []

        # Layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        if layerFunctions is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(TransferFunctions.linear)
                else:
                    lFuncs.append(TransferFunctions.sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Incompatible list of transfer functions.")
            elif layerFunctions[0] is not None:
                raise ValueError("Input layer cannot have a transfer function.")
            else:
                lFuncs = layerFunctions[1:]

        self.tFuncs = lFuncs

        # Data from last Run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        # Create the weight arrays
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.01, size=(l2, l1 + 1)))
            self._previousWeightDelta.append(np.zeros((l2, l1 + 1)))

    #
    # Run method
    #
    def Run(self, input):
        """Run the network based on the input data"""

        lnCases = input.shape[0]

        # Clear out the previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # Run it!
        for index in range(self.layerCount):
            # Determine layer input
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](TransferFunctions(), layerInput, False))

        return self._layerOutput[-1].T

    #
    # TrainEpoch method
    #
    def TrainEpoch(self, input, target, trainingRate=0.2, momentum=0.5):
        """This method trains the network for one epoch"""

        delta = []
        lnCases = input.shape[0]

        # First run the network
        self.Run(input)

        # Calculate our deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                # Compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.tFuncs[index](TransferFunctions(), self._layerInput[index], True))
            else:
                # Compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(
                    delta_pullback[:-1, :] * self.tFuncs[index](TransferFunctions(), self._layerInput[index], True))

        # Compute weight deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            curWeightDelta = np.sum( \
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0) \
                , axis=0)

            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]

            self.weights[index] -= weightDelta

            self._previousWeightDelta[index] = weightDelta

        return error


def read_digits_input(dirName):
    all_digit_arrays = []
    for filename in sorted(os.listdir(dirName)):
        print(filename)
        with open("./" + dirName + "/" + filename, "r") as myfile:
            data = myfile.read()
            input_data = data.replace("\n", "")

            one_digit_array = [1]
            i = 0
            while i < 35:
                if input_data[i] == "*":
                    one_digit_array.append(1)
                else:
                    one_digit_array.append(0)
                i += 1
            # print(input_data)
            # print(one_digit_array)
            all_digit_arrays.append(one_digit_array)

    return all_digit_arrays


def read_digits_output(dirName):
    all_output_arrays = []
    for filename in sorted(os.listdir(dirName)):
        print(filename)
        with open("./" + dirName + "/" + filename, "r") as myfile:
            data = myfile.read()
            output_data = data.replace("\n", "")

            one_digit_array = []
            i = 35
            while len(one_digit_array) < 10 and i < 60:
                if output_data[i] == "0":
                    one_digit_array.append(0)
                else:
                    if output_data[i] == "1":
                        one_digit_array.append(1)
                i += 1
            # print(output_data)
            # print(one_digit_array)
            all_output_arrays.append(one_digit_array)

    return all_output_arrays

#
# If run as a script, create a test object
#
if __name__ == "__main__":

    inputData = read_digits_input("trainingset")
    outputData = read_digits_output("trainingset")
    print(inputData)
    print(outputData)
    lvInput = np.array(inputData)
    lvTarget = np.array(outputData)
    lFuncs = [None, TransferFunctions.tanh, TransferFunctions.linear]

    bpn = BackPropagationNetwork((36, 36, 10), lFuncs)

    lnMax = 50000
    lnErr = 1e-6
    for i in range(lnMax + 1):
        err = bpn.TrainEpoch(lvInput, lvTarget, momentum=0.7)
        if i % 5000 == 0 and i > 0:
            print("Iteration {0:6d}K - Error: {1:0.6f}".format(int(i / 1000), err))
        if err <= lnErr:
            print("Desired error reached. Iter: {0}".format(i))
            break

    # Display output

    lvOutput = bpn.Run(lvInput)
    for i in range(lvInput.shape[0]):
        print("Input: {0} Output: {1}".format(lvInput[i], lvOutput[i]))
