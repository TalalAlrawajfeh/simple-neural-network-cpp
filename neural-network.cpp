#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <math.h>
#include <time.h>
#include <string>
#include <sstream>

class PropagationFunction {
public:
    virtual double propagate(double *inputs, int numberOfInputs, double *weights, double bias) = 0;
};

class ActivationFunction {
public:
    virtual double activate(double input) = 0;
};

class OutputFunction {
public:
    virtual double output(double activation) = 0;
};

class RandomWeightsGenerator {
public:
    virtual double nextRandomWeight() = 0;
};

class WeightedSumPropagationFunction : public PropagationFunction {
public:
    double propagate(double *inputs, int numberOfInputs, double *weights, double bias)
    {
        double sum = 0;
        for(int i = 0; i < numberOfInputs; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum + bias;
    }
};

class TanhActivationFunction : public ActivationFunction {
public:
    double activate(double input)
    {
        return tanh(input);
    }
};

class IdentityOutputFunction : public OutputFunction {
public:
    double output(double activation) 
    {
        return activation;
    }
};

class BipolarRandomWeightsGenerator : public RandomWeightsGenerator {
public:
    double nextRandomWeight() {
        return 1 - ((double) rand() / (double) RAND_MAX) * 2;
    }

    BipolarRandomWeightsGenerator() {
        srand(time(NULL));
    }
};

class TrainingUnit {
    double *inputs;
    int numberOfInputs;
    double expectedOutput;

public:
    TrainingUnit &setInputs(double *inputs)
    {
        this->inputs = inputs;
        return *this;
    }

    TrainingUnit &setNumberOfInputs(int numberOfInputs)
    {
        this->numberOfInputs = numberOfInputs;
        return *this;
    }

    TrainingUnit &setExpectedOutput(double expectedOutput) 
    {
        this->expectedOutput = expectedOutput;
        return *this;
    }

    double *getInputs() const
    {
        return inputs;
    }

    int getNumberOfInputs() const
    {
        return numberOfInputs;
    }

    double getExpectedOutput() const
    {
        return expectedOutput;
    }
};

class Neuron {
    int numberOfInputs;
    double *weights;
    double bias;

    WeightedSumPropagationFunction propagationFunction;
    TanhActivationFunction activationFunction;
    IdentityOutputFunction outputFunction;
    BipolarRandomWeightsGenerator randomWeightsGenerator;

    void initializeRandomWeights()
    {
        weights = new double[numberOfInputs];
        for(int i = 0; i < numberOfInputs; i++) {
            weights[i] = randomWeightsGenerator.nextRandomWeight();
        }
        bias = randomWeightsGenerator.nextRandomWeight();
    }
public:
    Neuron(int numberOfInputs)
    {
        this->numberOfInputs = numberOfInputs;
        initializeRandomWeights();
    }

    ~Neuron()
    {
        delete[] weights;
    }

    double feedForward(double *inputs)
    {
        return outputFunction.output(activationFunction.activate(propagationFunction.propagate(inputs, numberOfInputs, weights, bias)));
    }

    void train(TrainingUnit *trainingUnits, int numberOfTrainingUnits, double learningRate) {
        for(int i = 0; i < numberOfTrainingUnits; i++) {
            double *inputs = trainingUnits[i].getInputs();
            double error = trainingUnits[i].getExpectedOutput() - feedForward(inputs);
            for(int i = 0; i < numberOfInputs; i++) {
                weights[i] += learningRate * error * inputs[i];
            }
            bias += learningRate * error;
        }
    }
};	

void readTrainingData(char *filePath, TrainingUnit **trainingUnitsPtr, int *numberOfTrainingUnitsPtr)
{
    std::vector<TrainingUnit> trainingUnits;
    std::ifstream inputStream(filePath);
    *numberOfTrainingUnitsPtr = 0;

    while(!inputStream.eof()) {
        std::string buffer;
        getline(inputStream, buffer, '\n');
        std::stringstream ss(buffer);
        
        double value;
        std::vector<double> values;
        while(ss >> value) {
            values.push_back(value);
            char next = ss.peek();
            if(next == ',' || next == ' ') {
                ss.ignore();
            }
        }
        
        TrainingUnit trainingUnit;
        double *inputs = new double[1];
        inputs[0] = values.at(0);
        inputs[1] = values.at(1);
        trainingUnit.setInputs(inputs).setNumberOfInputs(2).setExpectedOutput(values.at(2));
        trainingUnits.push_back(trainingUnit);

        *numberOfTrainingUnitsPtr += 1;
    }

    *trainingUnitsPtr = new TrainingUnit[*numberOfTrainingUnitsPtr];
    std::copy(trainingUnits.begin(), trainingUnits.end(), *trainingUnitsPtr);

    inputStream.close();
}

int main(int argc, char **argv) {
    TrainingUnit *trainingUnits;
    int numberOfTrainingUnits;
    std::string filePath = "./training_data.csv\0";
    readTrainingData((char *) filePath.c_str(), &trainingUnits, &numberOfTrainingUnits);
    Neuron neuron(2);
    neuron.train(trainingUnits, numberOfTrainingUnits, 0.03);

    double *inputs1 = new double[1];
    inputs1[0] = -2;
    inputs1[1] = 2;

    double *inputs2 = new double[1];
    inputs2[0] = 4;
    inputs2[1] = -4;

    std::cout << neuron.feedForward(inputs1) << std::endl;
    std::cout << neuron.feedForward(inputs2) << std::endl;
    
    delete[] inputs1;
    delete[] inputs2;
    delete[] trainingUnits;
    
    return 0;
}
