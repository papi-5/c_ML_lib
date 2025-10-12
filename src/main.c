#include <stdio.h>
#include "machl_export.h"

int main ()
{
	printf ("helo\n");
/*	int arr[] = {4, 4, 2, 2};
	int num = 4;
	NeuralNet *net = createNeuralNet (num, arr);
	initializeNeuralNet (net, -2, 2, -5, 5);
	int ar[] = {0, 0, 3};
	setActivationFunctions (net, ar);
	setCostFunction (net, 1);
	setAlpha (net, 0.1);

	double a[] = {0, 0, 0, 0, 0, 1,
			0, 0, 0, 1, 1, 0,
			0, 0, 1, 0, 1, 0,
			0, 0, 1, 1, 0, 1,
			0, 1, 0, 0, 1, 0,
			0, 1, 0, 1, 0, 1,
			0, 1, 1, 0, 0, 1,
			0, 1, 1, 1, 1, 0,
			1, 0, 0, 0, 1, 0,
			1, 0, 0, 1, 0, 1,
			1, 0, 1, 0, 0, 1,
			1, 0, 1, 1, 1, 0,
			1, 1, 0, 0, 0, 1,
			1, 1, 0, 1, 1, 0,
			1, 1, 1, 0, 1, 0,
			1, 1, 1, 1, 0, 1};
	int b[] = {4, 4, 4, 4};
	double cost;
	for (int i = 0; i < 100000; i++) {
		if (i % 10000 == 0) {
			cost = neuralNetCost (net, 16, a);
			printf ("%lf\n", cost);
			printMatrix (net -> layers[2] -> output);
		}
		trainNeuralNet (net, 4, b, a);
	}
*/	return 0;
}
