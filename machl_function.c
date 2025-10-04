#include <math.h>
#include "machl_tensor.h"
#include "machl_function.h"

ActFunction activationFunctions[] = {{sigmoid, derSigmoid},
									 {Tanh, derTanh},
									 {ReLU, derReLU},
									 {softmax, derSoftmax}};

CostFunction costFunctions[] = {{MSE, derMSE},
								{crossEntropy, derCrossEntropy}};

void sigmoid (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		ten -> ten[i] = 1.0 / (1.0 + exp (ten -> ten[i] * -1));
}

void derSigmoid (Tensor *ten, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++) {
		float x = ten -> ten[i];
		res -> ten[i] = x * (1.0 - x);
	}
}

void Tanh (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++) {
		float x = ten -> ten[i];
		float x1 = exp (x);
		float x2 = exp (-x);
		ten ->ten[i] = (x1 - x2) / (x1 + x2);
	}
}

void derTanh (Tensor *ten, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		res -> ten[i] = 1.0 - pow (ten ->ten[i], 2.0);
}

void ReLU (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++) {
		if (ten -> ten[i] < 0)
			ten -> ten[i] = 0;
	}
}

void derReLU (Tensor *ten, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++) {
		res -> ten[i] = (ten -> ten[i] > 0) ? 1 : 0;
	}
}

void softmax (Tensor *ten)
{
	int length = ten -> rows * ten -> coll;
	float sum = 0;
	Tensor *tmp = createTensor (1, length);

	for (int i = 0; i < length; i++) {
		tmp -> ten[i] = exp (tmp -> ten[i]);
		sum += tmp -> ten[i];
	}

	for (int i = 0; i < length; i++)
		ten -> ten[i] = tmp -> ten[i] / sum;

	deleteTensor (tmp);
}

void derSoftmax (Tensor *ten, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++) {
		res -> ten[i] = 0;
		float x = ten -> ten[i];
		for (int j = 0; j < length; j++) {
			float y = (i == j) ? 1 - x : ten -> ten[j] * -1;
			res -> ten[i] += x * y;
		}
	}
}

float MSE (Tensor *ten, Tensor *y)
{
	int length = ten -> rows * ten -> coll;
	float res = 0;

	for (int i = 0; i < length; i++) {
		float tmp = ten -> ten[i] - y -> ten[i];
		res += pow (tmp, 2.0);
	}

	res /= length;

	return res;
}

void derMSE (Tensor *ten, Tensor *y, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		res -> ten[i] = 2.0 * (ten -> ten[i] - y -> ten[i]);
}

float crossEntropy (Tensor *ten, Tensor *y)
{
	int length = ten -> rows * ten -> coll;
	float res = 0;

	for (int i = 0; i < length; i++)
		res += y -> ten[i] * log (ten -> ten[i]);

	res *= -1.0;

	return res;
}

void derCrossEntropy (Tensor *ten, Tensor *y, Tensor *res)
{
	int length = ten -> rows * ten -> coll;

	for (int i = 0; i < length; i++)
		res -> ten[i] = -1.0 * (y -> ten[i] / ten -> ten[i]);
}
