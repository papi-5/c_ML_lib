#include <stdio.h>
#include "mcl_io.h"

int main ()
{
	int neurons[4] = {784, 128, 64, 10};
	mcl_network *net = mcl_network_create (4, neurons);

	mcl_network_init_xavier_normal (net);
	//mcl_network_print (net);
	mcl_network_print_meta (net);
	printf ("network size: %ld bytes\n\n", mcl_network_size (net));

	mcl_tensor *ten1 = mcl_tensor_create (2, 2);
	mcl_tensor *ten2 = mcl_tensor_create (2, 2);
	mcl_tensor *ten3 = mcl_tensor_create (2, 2);
	for (int i = 0; i < 4; i++) {
		ten1 -> ten[i] = i;
	}

	mcl_tensor_print (ten1);
	mcl_tensor_transpose (ten1, ten2);
	mcl_tensor_print (ten2);
	mcl_tensor_multiply (ten1, ten2, ten3);
	mcl_tensor_print (ten3);


/*	int arr[] = {4, 4, 2, 2};
	int num = 4;
	NeuralNet *net = createNeuralNet (num, arr);
	initializeNeuralNet (net, -2, 2, -5, 5);
	int ar[] = {0, 0, 3};
	setActivationFunctions (net, ar);
	mcl_network_set_cost (net, 1);
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
