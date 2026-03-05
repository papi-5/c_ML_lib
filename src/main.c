#include <stdio.h>
#include "mcl_io.h"
#include "mcl_dataset.h"

void test_tensors ()
{
	mcl_tensor *ten1 = mcl_tensor_create (35, 34);
	mcl_tensor *ten2 = mcl_tensor_create (34, 33);
	mcl_tensor *ten3 = mcl_tensor_create (34, 35);
	mcl_tensor *ten4 = mcl_tensor_create (35, 33);
	mcl_tensor *ten5 = mcl_tensor_create (35, 33);

	mcl_tensor_random_normal (ten1);
	mcl_tensor_random_normal (ten2);
	mcl_tensor_transpose (ten1, ten3);

	//mcl_tensor_print (ten1);
	//mcl_tensor_print (ten2);
	mcl_tensor_mul (ten1, ten2, ten4);
	mcl_tensor_multiply (ten1, ten2, ten5);
	for (int i = 0; i < 35*33; i++) {
		if (ten4 -> ten[i] != ten5 -> ten[i]) {
			printf ("wrong\n");
		}
	}
	mcl_tensor_reset (ten4);
	mcl_tensor_mul_t (ten3, ten2, ten4);
	for (int i = 0; i < 35*33; i++) {
		if (ten4 -> ten[i] != ten5 -> ten[i]) {
			printf ("wrong\n");
		}
	}
	mcl_tensor_print (ten4);
}

void test_io ()
{
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
}

void test_dataset ()
{
	void data_print (mcl_tensor **data, int size)
	{
		for (int i = 0; i < size * 2; i++) {
			mcl_tensor_print (data[i]);
		}
	}

	mcl_dataset *data0 = mcl_dataset_create (4, 2, 0);
	mcl_dataset *data1 = mcl_dataset_create (4, 2, 1);
	mcl_dataset_load_train (data0, "test0.csv");
	data_print (data0 -> train, data0 -> train_size);
	mcl_dataset_load_test (data0, "test0.csv");
	data_print (data0 -> test, data0 -> test_size);
	mcl_dataset_load_split (data1, "test1.csv", 0.8);
	data_print (data1 -> test, data1 -> test_size);
}

int main ()
{
	test_tensors ();
	return 0;
}
