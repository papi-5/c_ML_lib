#ifndef MACHL_FUNC_H
#define MACHL_FUNC_H

#include "machl_tensor.h"

struct actFunction {
	void (*function) (Tensor*);
	void (*derFunction) (Tensor*, Tensor*);
};

typedef struct actFunction ActFunction;

struct costFunction {
	float (*function) (Tensor*, Tensor*);
	void (*derFunction) (Tensor*, Tensor*, Tensor*);
};

typedef struct costFunction CostFunction;

void sigmoid (Tensor *ten);

void derSigmoid (Tensor *ten, Tensor *res);

void Tanh (Tensor *ten);

void derTanh (Tensor *ten, Tensor *res);

void ReLU (Tensor *ten);

void derReLU (Tensor *ten, Tensor *res);

void softmax (Tensor *ten);

void derSoftmax (Tensor *ten, Tensor *res);

float MSE (Tensor *ten, Tensor *y);

void derMSE (Tensor *ten, Tensor *y, Tensor *res);

float crossEntropy (Tensor *ten, Tensor *y);

void derCrossEntropy (Tensor *ten, Tensor *y, Tensor *res);

#endif
