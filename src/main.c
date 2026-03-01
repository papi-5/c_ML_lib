#include <stdio.h>
#include "mcl_io.h"
#include "mcl_dataset.h"

int main ()
{
/*
	int neurons[4] = {4, 4, 2, 2};
	mcl_network *net = mcl_network_create (4, neurons);

	mcl_network_init_xavier_normal (net);
	mcl_network_print (net);
	mcl_network_print_meta (net);
	printf ("network size: %ld bytes\n\n", mcl_network_size (net));

	mcl_network_export (net, "test.mcl");
	mcl_network *net2 = mcl_network_import ("test.mcl");
	mcl_network_print (net2);
	mcl_network_print_meta (net2);
	printf ("network size: %ld bytes\n\n", mcl_network_size (net));
*/
/*
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
*/
	void data_print (mcl_tensor **data, int size)
	{
		for (int i = 0; i < size * 2; i++) {
			mcl_tensor_print (data[i]);
		}
	}

	mcl_dataset *data0 = mcl_dataset_create (4, 2, 0);
	mcl_dataset *data1 = mcl_dataset_create (4, 2, 1);
	mcl_dataset_load_train (data0, "test0.csv");
	//data_print (data0 -> train, data0 -> train_size);
	mcl_dataset_load_test (data0, "test0.csv");
	//data_print (data0 -> test, data0 -> test_size);
	mcl_dataset_load_split (data1, "test1.csv", 0.8);
	printf ("%d\n\n", data1 -> test[1]);
	data_print (data1 -> test, data1 -> test_size);

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
