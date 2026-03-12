#include <math.h>
#include <stdlib.h>
#include "mcl_tensor.h"
#include "mcl_function.h"

mcl_activation activation_functions[] = {{mcl_sigmoid, mcl_sigmoid_d},
									 {mcl_tanh, mcl_tanh_d},
									 {mcl_relu, mcl_relu_d},
									 {mcl_softmax, mcl_softmax_d}};

mcl_cost cost_functions[] = {{mcl_mse, mcl_mse_d},
								{mcl_cross_entropy, mcl_cross_entropy_d}};

void mcl_sigmoid (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		ten -> ten[i] = 1.0 / (1.0 + exp (ten -> ten[i] * -1));
}

void mcl_sigmoid_d (mcl_tensor *ten, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++) {
		float x = ten -> ten[i];
		res -> ten[i] = x * (1.0 - x);
	}
}

void mcl_tanh (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++) {
		float x = ten -> ten[i];
		float x1 = exp (x);
		float x2 = exp (-x);
		ten ->ten[i] = (x1 - x2) / (x1 + x2);
	}
}

void mcl_tanh_d (mcl_tensor *ten, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		res -> ten[i] = 1.0 - pow (ten ->ten[i], 2.0);
}

void mcl_relu (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++) {
		if (ten -> ten[i] < 0)
			ten -> ten[i] = 0;
	}
}

void mcl_relu_d (mcl_tensor *ten, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++) {
		res -> ten[i] = (ten -> ten[i] > 0) ? 1 : 0;
	}
}

void mcl_softmax (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;
	float sum = 0;
	float max = ten -> ten[0];
	float *tmp = malloc (length * sizeof (float));

	for (int i = 1; i < length; i++) {
		if (max < ten -> ten[i]) {
			max = ten -> ten[i];
		}
	}

	for (int i = 0; i < length; i++) {
		tmp[i] = exp (ten -> ten[i] - max);
		sum += tmp[i];
	}

	for (int i = 0; i < length; i++)
		ten -> ten[i] = tmp[i] / sum;

	free (tmp);
}

void mcl_softmax_d (mcl_tensor *ten, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++) {
		res -> ten[i] = 0;
		float x = ten -> ten[i];
		for (int j = 0; j < length; j++) {
			float y = (i == j) ? 1 - x : ten -> ten[j] * -1;
			res -> ten[i] += x * y;
		}
	}
}

float mcl_mse (mcl_tensor *ten, mcl_tensor *y)
{
	int length = ten -> row * ten -> col;
	float res = 0;

	for (int i = 0; i < length; i++) {
		float tmp = ten -> ten[i] - y -> ten[i];
		res += pow (tmp, 2.0);
	}

	res /= length;

	return res;
}

void mcl_mse_d (mcl_tensor *ten, mcl_tensor *y, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		res -> ten[i] = 2.0 * (ten -> ten[i] - y -> ten[i]);
}

float mcl_cross_entropy (mcl_tensor *ten, mcl_tensor *y)
{
	int length = ten -> row * ten -> col;
	float res = 0;

	for (int i = 0; i < length; i++)
		res += y -> ten[i] * log (ten -> ten[i]);

	res *= -1.0;

	return res;
}

void mcl_cross_entropy_d (mcl_tensor *ten, mcl_tensor *y, mcl_tensor *res)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		res -> ten[i] = -1.0 * (y -> ten[i] / ten -> ten[i]);
}
