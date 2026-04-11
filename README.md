# C ML library

This is a machine learning library for C.

## Features

- multi layer perceptron specification
    - number of neurons per layer
    - activation functions per layer
        - sigmoid, tanh, ReLU, softmax (used only with cross entropy), linear
    - xavier and kaiming initialization
- dataset specification
    - loading and splitting a CSV file into train and test sets
- optimizer specification
    - loss function
        - MSE, cross entropy
    - learn rate, dropout, beta1, beta2, epsilon
    - SGD and ADAM optimizer
- utilities
    - export and import ANNs

## Requirements

- gcc
- make
- x86 intel

## Building

make

## Usage

```c
#include <stdio.h>
#include "mcl.h"

int main ()
{
    int input_size = 784;
    int output_size = 10;

    mcl_dataset *data = mcl_dataset_create (MCL_CLASSIFICATION, MCL_FIRST, input_size, output_size);

	mcl_dataset_load_train (data, "dataset/mnist_train.csv");
	mcl_dataset_load_test (data, "dataset/mnist_test.csv");

    // neurons include input, output and the hidden layers
    int neurons[] = {784, 128, 64, 10};
    int neurons_size = 4;

    // activations are for every layer except the first
	mcl_activation_type activation[] = {MCL_RELU, MCL_RELU, MCL_SOFTMAX};

	mcl_network *net = mcl_network_create (neurons, neurons_size);
	mcl_network_set_activations (net, activation);
	mcl_network_init_kaiming (net);

    mcl_optimizer *opt = mcl_optimizer_create ();
    mcl_optimizer_set_dataset (opt, data);
	mcl_optimizer_set_network (opt, net);

    float learn_rate = 0.0001;
    float dropout = 0.05;

    mcl_optimizer_set_cost (opt, MCL_CROSS_ENTROPY);
    mcl_optimizer_set_learn_rate (opt, learn_rate);
    mcl_optimizer_set_dropout (opt, dropout);

    float acc, cost;
    int batch_size = 100;
    int epochs = 2;
	
	cost = mcl_optimizer_test_train (opt, batch_size, &acc);
	printf ("cost: %f acc: %f\n\n", cost, acc);

	for (int i = 0; i < 10; i++) {
	    mcl_optimizer_train_adam (opt, batch_size, epochs);
		cost = mcl_optimizer_test_train (opt, batch_size, &acc);
		printf ("epochs: %d\n", (i + 1) * epochs);
		printf ("cost: %f acc: %f\n\n", cost, acc);
	}
    printf ("=====\n\n");

	printf ("test\n");
	cost = mcl_optimizer_test (opt, batch_size, &acc);
	printf ("cost: %f acc: %f\n\n", cost, acc);
	printf ("=====\n\n");

    mcl_network_export (net, "trained_nets/MNIST.mcl");

    return 0;
}
```

## Dataset format

The library can parse CSV files containing only floats.  
It does not support column names, named labels and sample index parsing.

## Limitations

- only MLP architectures
- specific CSV format
    - only floats, no strings
- no GPU implementation for now
- only SIMD on x86 arch

## Datasets used

4-bit XOR (manually made)  
polynomial regression (made with generate_regression.py)  
[Iris](https://www.kaggle.com/datasets/saurabh00007/iriscsv) (cleaned up with clean_iris.py)  
[MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
[Boston housing](https://www.kaggle.com/datasets/kyasar/boston-housing)
