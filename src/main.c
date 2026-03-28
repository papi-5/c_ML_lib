#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mcl_io.h"
#include "mcl_optimizer.h"

void test_tensors ()
{
	mcl_tensor *ten1 = mcl_tensor_create (35, 34);
	mcl_tensor *ten2 = mcl_tensor_create (34, 33);
	mcl_tensor *ten1t = mcl_tensor_create (34, 35);
	mcl_tensor *ten2t = mcl_tensor_create (33, 34); 
	mcl_tensor *ten4 = mcl_tensor_create (35, 33);
	mcl_tensor *ten5 = mcl_tensor_create (35, 33);

	mcl_tensor_random_normal (ten1);
	mcl_tensor_random_normal (ten2);
	mcl_tensor_transpose (ten1, ten1t);
	mcl_tensor_transpose (ten2, ten2t);

	//mcl_tensor_print (ten1);
	//mcl_tensor_print (ten2);
	mcl_tensor_mul (ten1, ten2, ten4);
	mcl_tensor_multiply (ten1, ten2, ten5);
	for (int i = 0; i < 35*33; i++) {
		if (fabs (ten4 -> ten[i] - ten5 -> ten[i]) > 1e-4) {
			printf ("wrong\n");
			break;
		}
	}
	mcl_tensor_reset (ten4);
	mcl_tensor_mul_tl (ten1t, ten2, ten4);
	for (int i = 0; i < 35*33; i++) {
		if (fabs (ten4 -> ten[i] - ten5 -> ten[i]) > 1e-4) {
			printf ("wrong\n");
			break;
		}
	}
	mcl_tensor_reset (ten4);
	mcl_tensor_mul_tr (ten1, ten2t, ten4);
	for (int i = 0; i < 35*33; i++) {
		if (fabs (ten4 -> ten[i] - ten5 -> ten[i]) > 1e-4) {
			printf ("wrong\n");
			break;
		}
	}
	//mcl_tensor_print (ten4);
}

void test_multiplication_speed ()
{
	mcl_tensor *tena = mcl_tensor_create (1000, 5000);
	mcl_tensor *tenb = mcl_tensor_create (5000, 1000);
	mcl_tensor *res = mcl_tensor_create (1000, 1000);
	clock_t t;

	mcl_tensor_random_normal (tena);
	mcl_tensor_random_normal (tenb);
	printf ("1000x5000x1000\n");

	t = clock ();
	mcl_tensor_multiply (tena, tenb, res);
	t = clock () - t;
	printf ("time naive: %f\n", ((double)t) / CLOCKS_PER_SEC);

	t = clock();
	mcl_tensor_mul (tena, tenb, res);
	t = clock () - t;
	printf ("time tiled: %f\n", ((double)t) / CLOCKS_PER_SEC);

	mcl_tensor_delete (tena);
	mcl_tensor_delete (tenb);
	mcl_tensor_delete (res);

	tena = mcl_tensor_create (512, 784);
	tenb = mcl_tensor_create (784, 512);
	res = mcl_tensor_create (512, 512);

	mcl_tensor_random_normal (tena);
	mcl_tensor_random_normal (tenb);
	printf ("512x784x512\n");

	t = clock ();
	mcl_tensor_multiply (tena, tenb, res);
	t = clock () - t;
	printf ("time naive: %f\n", ((double)t) / CLOCKS_PER_SEC);

	t = clock();
	mcl_tensor_mul (tena, tenb, res);
	t = clock () - t;
	printf ("time tiled: %f\n", ((double)t) / CLOCKS_PER_SEC);
}

void test_io ()
{
	int neurons[4] = {4, 4, 2, 2};
	mcl_network *net = mcl_network_create (neurons, 4);

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

	mcl_dataset *data0 = mcl_dataset_create (MCL_CLASSIFICATION, MCL_FIRST, 4, 2);
	mcl_dataset *data1 = mcl_dataset_create (MCL_CLASSIFICATION, MCL_LAST, 4, 2);
	mcl_dataset_load_train (data0, "dataset/test0.csv");
	data_print (data0 -> train, data0 -> train_size);
	mcl_dataset_load_test (data0, "dataset/test0.csv");
	data_print (data0 -> test, data0 -> test_size);
	mcl_dataset_load_split (data1, "dataset/test1.csv", 0.8);
	data_print (data1 -> test, data1 -> test_size);
}

void test_forward ()
{
	int neurons[] = {4, 4, 2, 2};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_init_kaiming (net);
	mcl_tensor *input = mcl_tensor_create (4, 1);
	mcl_tensor_random_normal (input);
	mcl_network_forward_test (net, input);
	mcl_network_print (net);
	mcl_tensor_print (input);
	mcl_tensor_print (net -> layers[2] -> output);
}

void test_sgd ()
{
	int neurons[] = {4, 16, 16, 2};
	mcl_activation_type activation[] = {MCL_TANH, MCL_TANH, MCL_SIGMOID};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_set_activations (net, activation);
	mcl_network_init_xavier_normal (net);

	//mcl_network_print (net);
	//mcl_network_print_meta (net);
	//mcl_network_print_grad (net);

	mcl_dataset *data = mcl_dataset_create (MCL_CLASSIFICATION, MCL_FIRST, 4, 2);
	mcl_dataset_load_train (data, "dataset/test0.csv");
	mcl_dataset_load_test (data, "dataset/test0.csv");

	int train_size = mcl_dataset_train_samples (data);
	int test_size = mcl_dataset_test_samples (data);
	printf ("train samples: %d test samples: %d\n\n", train_size, test_size);

	mcl_optimizer *opt = mcl_optimizer_create ();
	mcl_optimizer_set_dataset (opt, data);
	mcl_optimizer_set_network (opt, net);
	mcl_optimizer_set_learn_rate (opt, 0.2);

	float acc;
	float loss;
	loss = mcl_optimizer_test_train (opt, 16, &acc);
	printf ("loss: %f acc: %f\n\n", loss, acc);
	mcl_optimizer_train_sgd (opt, 16, 1);
	loss = mcl_optimizer_test_train (opt, 16, &acc);
	printf ("loss: %f acc: %f\n\n", loss, acc);
	for (int i = 0; i < 10; i++) {
		mcl_optimizer_train_sgd (opt, 16, 200);
		loss = mcl_optimizer_test_train (opt, 16, &acc);
		printf ("%d\n", i);
		printf ("loss: %f acc: %f\n\n", loss, acc);
	}
	loss = mcl_optimizer_test (opt, 16, &acc);
	printf ("test\nloss: %f acc: %f\n\n", loss, acc);
}

int main ()
{
	test_sgd ();
	return 0;
}
