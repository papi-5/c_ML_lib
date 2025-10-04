#ifndef MACHL_LAYER_H
#define MACHL_LAYER_H

#include "machl_tensor.h"
#include "machl_function.h"

struct layer {
	Tensor *weights;
	Tensor *weightsT;
	Tensor *biases;
	Tensor *output;
	Tensor *outputT;
	Tensor *weightG;
	Tensor *biasG;
	Tensor *inputG;
	Tensor *costG;
	ActFunction *actFunction;
};

typedef struct layer Layer;

Layer* createLayer (int rows, int coll);

void uRandomWeights (Layer *lay, float lowerLimit, float upperLimit);

void nRandomWeights (Layer *lay, float mean, float stddev);

void uRandomBiases (Layer *lay, float lowerLimit, float upperLimit);

void nRandomBiases (Layer *lay, float mean, float stddev);

void printLayer (Layer *lay);

void deleteLayer (Layer *lay);

#endif
