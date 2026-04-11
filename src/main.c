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
	mcl_dataset_load_train (data0, "dataset/xor4_0.csv");
	//data_print (data0 -> train, data0 -> train_size);
	mcl_dataset_load_test (data0, "dataset/xor4_0.csv");
	//data_print (data0 -> test, data0 -> test_size);
	mcl_dataset_load_split (data1, "dataset/xor4_1.csv", 0.8);
	//data_print (data1 -> test, data1 -> test_size);

	mcl_dataset *data_reg_0 = mcl_dataset_create (MCL_REGRESSION, MCL_FIRST, 1, 1);
	mcl_dataset_load_split (data_reg_0, "dataset/regression_0.csv", 0.995);
	printf ("train size: %d test size: %d\n\n", data_reg_0 -> train_size, data_reg_0 -> test_size);
	data_print (data_reg_0 -> test, data_reg_0 -> test_size);
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

mcl_optimizer* get_opt (mcl_network *net, mcl_dataset *data)
{
	mcl_optimizer *opt = mcl_optimizer_create ();
	mcl_optimizer_set_dataset (opt, data);
	mcl_optimizer_set_network (opt, net);

	return opt;
}

mcl_network* get_net_xor ()
{
	int neurons[] = {4, 16, 16, 2};
	mcl_activation_type activation[] = {MCL_TANH, MCL_TANH, MCL_SIGMOID};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_set_activations (net, activation);
	mcl_network_init_xavier_normal (net);

	return net;
}

mcl_network* get_net_reg ()
{
	int neurons[] = {1, 16, 16, 1};
	mcl_activation_type activation[] = {MCL_TANH, MCL_TANH, MCL_LINEAR};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_set_activations (net, activation);
	mcl_network_init_xavier_normal (net);

	return net;
}

mcl_network* get_net_iris ()
{
	int neurons[] = {4, 8, 3};
	mcl_activation_type activation[] = {MCL_TANH, MCL_SOFTMAX};
	mcl_network *net = mcl_network_create (neurons, 3);
	mcl_network_set_activations (net, activation);
	mcl_network_init_kaiming (net);

	return net;
}

mcl_network* get_net_mnist ()
{
	int neurons[] = {784, 128, 64, 10};
	mcl_activation_type activation[] = {MCL_RELU, MCL_RELU, MCL_SOFTMAX};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_set_activations (net, activation);
	mcl_network_init_kaiming (net);

	return net;
}

mcl_network* get_net_boston ()
{
	int neurons[] = {13, 64, 32, 1};
	mcl_activation_type activation[] = {MCL_RELU, MCL_RELU, MCL_LINEAR};
	mcl_network *net = mcl_network_create (neurons, 4);
	mcl_network_set_activations (net, activation);
	mcl_network_init_kaiming (net);

	return net;
}

mcl_dataset* get_data_xor ()
{
	mcl_dataset *data = mcl_dataset_create (MCL_CLASSIFICATION, MCL_FIRST, 4, 2);
	mcl_dataset_load_train (data, "dataset/xor4_0.csv");
	mcl_dataset_load_test (data, "dataset/xor4_0.csv");

	return data;
}

mcl_dataset* get_data_reg ()
{
	mcl_dataset *data = mcl_dataset_create (MCL_REGRESSION, MCL_FIRST, 1, 1);
	mcl_dataset_load_split (data, "dataset/regression_0.csv", 0.8);

	return data;
}

mcl_dataset* get_data_iris ()
{
	mcl_dataset *data = mcl_dataset_create (MCL_CLASSIFICATION, MCL_LAST, 4, 3);
	mcl_dataset_load_split (data, "dataset/iris_clean.csv", 0.8);

	return data;
}

mcl_dataset* get_data_mnist ()
{
	mcl_dataset *data = mcl_dataset_create (MCL_CLASSIFICATION, MCL_FIRST, 784, 10);
	//mcl_dataset_load_split (data, "dataset/mnist_train.csv", 0.1);
	mcl_dataset_load_train (data, "dataset/mnist_train.csv");
	mcl_dataset_load_test (data, "dataset/mnist_test.csv");

	return data;
}
mcl_dataset* get_data_boston ()
{
	mcl_dataset *data = mcl_dataset_create (MCL_REGRESSION, MCL_LAST, 13, 1);
	mcl_dataset_load_split (data, "dataset/boston_housing.csv", 0.8);

	return data;
}

void run_test (mcl_optimizer *opt, float learn_rate, int batch_size, int epochs, int algorithm)
{
	mcl_optimizer_set_learn_rate (opt, learn_rate);

	int train_size = mcl_dataset_train_samples (opt -> data);
	int test_size = mcl_dataset_test_samples (opt -> data);
	printf ("train samples: %d test samples: %d\n", train_size, test_size);

	int input_size = mcl_dataset_input_size (opt -> data);
	int output_size = mcl_dataset_output_size (opt -> data);
	printf ("input size: %d output size: %d\n", input_size, output_size);

	if (algorithm)
		printf ("optimization algorithm: ADAM\n");
	else
		printf ("optimization algorithm: SGD\n");

	printf ("\n");
	printf ("=====\n\n");

	float acc, cost;
	
	cost = mcl_optimizer_test_train (opt, batch_size, &acc);
	printf ("cost: %f acc: %f\n\n", cost, acc);
	for (int i = 0; i < 10; i++) {
		if (algorithm == 0)
			mcl_optimizer_train_sgd (opt, batch_size, epochs);
		else
			mcl_optimizer_train_adam (opt, batch_size, epochs);
		cost = mcl_optimizer_test_train (opt, batch_size, &acc);
		printf ("%d\n", (i + 1) * epochs);
		printf ("cost: %f acc: %f\n\n", cost, acc);
	}
	printf ("=====\n\n");

	printf ("test\n");
	cost = mcl_optimizer_test (opt, batch_size, &acc);
	printf ("cost: %f acc: %f\n\n", cost, acc);
	printf ("=====\n\n");
}

void test_xor_sgd ()
{
	printf ("4-BIT XOR\n");
	mcl_network *net = get_net_xor ();
	mcl_dataset *data = get_data_xor ();
	mcl_optimizer *opt = get_opt (net, data);
	run_test (opt, 0.2, 16, 250, 0);
}

void test_xor_adam ()
{
	printf ("4-BIT XOR\n");
	mcl_network *net = get_net_xor ();
	mcl_dataset *data = get_data_xor ();
	mcl_optimizer *opt = get_opt (net, data);
	run_test (opt, 0.001, 16, 25, 1);
}

void test_reg_sgd ()
{
	printf ("SIMPLE REGRESSION\n");
	mcl_network *net = get_net_reg ();
	mcl_dataset *data = get_data_reg ();
	mcl_optimizer *opt = get_opt (net, data);
	run_test (opt, 0.01, 100, 100, 0);
}

void test_reg_adam ()
{
	printf ("SIMPLE REGRESSION\n");
	mcl_network *net = get_net_reg ();
	mcl_dataset *data = get_data_reg ();
	mcl_optimizer *opt = get_opt (net, data);
	run_test (opt, 0.001, 100, 10, 1);
}

void test_iris_sgd ()
{
	printf ("IRIS\n");
	mcl_network *net = get_net_iris ();
	mcl_dataset *data = get_data_iris ();
	mcl_optimizer *opt = get_opt (net, data);
	mcl_optimizer_set_cost (opt, MCL_CROSS_ENTROPY);
	run_test (opt, 0.01, 50, 20, 0);
}

void test_iris_adam ()
{
	printf ("IRIS\n");
	mcl_network *net = get_net_iris ();
	mcl_dataset *data = get_data_iris ();
	mcl_optimizer *opt = get_opt (net, data);
	mcl_optimizer_set_cost (opt, MCL_CROSS_ENTROPY);
	run_test (opt, 0.001, 50, 2, 1);
}

void test_mnist_adam ()
{
	printf ("MNIST\n");
	mcl_network *net = get_net_mnist ();
	mcl_dataset *data = get_data_mnist ();
	mcl_optimizer *opt = get_opt (net, data);
	mcl_optimizer_set_cost (opt, MCL_CROSS_ENTROPY);
	run_test (opt, 0.0001, 100, 2, 1);
}

void test_boston_adam ()
{
	printf ("BOSTON HOUSING\n");
	mcl_network *net = get_net_boston ();
	mcl_dataset *data = get_data_boston ();
	mcl_optimizer *opt = get_opt (net, data);
	run_test (opt, 0.0001, 32, 50, 1);
}

int main ()
{
	printf ("\n");

	//test_tensors ();
	//test_multiplication_speed ();
	//test_io ();
	//test_dataset ();
	//test_forward ();
	
	//test_xor_sgd ();
	//test_xor_sgd ();
	//test_reg_sgd ();
	//test_reg_adam ();
	//test_iris_sgd ();
	//test_iris_adam ();
	//test_mnist_adam ();
	test_boston_adam ();

	return 0;
}
