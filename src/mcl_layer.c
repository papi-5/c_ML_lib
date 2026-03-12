#include <stdlib.h>
#include "mcl_tensor.h"
#include "mcl_layer.h"

mcl_layer* mcl_layer_create (int row, int col)
{
	mcl_layer *lay = malloc (sizeof (mcl_layer));

	lay -> weights = mcl_tensor_create (row, col);
	lay -> biases = mcl_tensor_create (row, 1);
	lay -> output = mcl_tensor_create (row, 1);
	lay -> weight_grad = mcl_tensor_create (row, col);
	lay -> bias_grad = mcl_tensor_create (row, 1);
	lay -> delta = mcl_tensor_create (row, 1);
	lay -> output_grad = mcl_tensor_create (row, 1);

	return lay;
}

size_t mcl_layer_size (mcl_layer *lay)
{
	size_t size = mcl_tensor_size(lay -> weights);
	size += mcl_tensor_size(lay -> biases);
	size += mcl_tensor_size(lay -> output);
	size += mcl_tensor_size(lay -> weight_grad);
	size += mcl_tensor_size(lay -> bias_grad);
	size += mcl_tensor_size(lay -> delta);
	size += mcl_tensor_size(lay -> output_grad);
	size += sizeof (mcl_layer);

	return size;
}

void mcl_layer_weights_rand_uniform (mcl_layer *lay, float lower, float upper)
{
	mcl_tensor_random_uniform (lay -> weights);
	mcl_tensor_scale (lay -> weights, upper - lower);
	mcl_tensor_add_scalar (lay -> weights, lower);
}

void mcl_layer_weights_rand_normal (mcl_layer *lay, float mean, float stddev)
{
	mcl_tensor_random_normal (lay -> weights);
	mcl_tensor_scale (lay -> weights, stddev);
	mcl_tensor_add_scalar (lay -> weights, mean);
}

void mcl_layer_biases_rand_uniform (mcl_layer *lay, float lower, float upper)
{
	mcl_tensor_random_uniform (lay -> biases);
	mcl_tensor_scale (lay -> biases, upper - lower);
	mcl_tensor_add_scalar (lay -> biases, lower);
}

void mcl_layer_biases_rand_normal (mcl_layer *lay, float mean, float stddev)
{
	mcl_tensor_random_normal (lay -> biases);
	mcl_tensor_scale (lay -> biases, stddev);
	mcl_tensor_add_scalar (lay -> biases, mean);
}

void mcl_layer_print (mcl_layer *lay)
{
	mcl_tensor_print (lay -> weights);
	mcl_tensor_print (lay -> biases);
}

void mcl_layer_reset_grad (mcl_layer *lay)
{
	mcl_tensor_reset (lay -> weight_grad);
	mcl_tensor_reset (lay -> bias_grad);
}

void mcl_layer_forward_train (mcl_layer *lay, mcl_tensor *input, float dropout)
{
	mcl_tensor_reset (lay -> output);
	mcl_tensor_mul (lay -> weights, input, lay -> output);
	mcl_tensor_add (lay -> output, lay -> biases);
	lay -> activation -> function (lay -> output);
	mcl_tensor_dropout (lay -> output, dropout);
}

void mcl_layer_forward_test (mcl_layer *lay, mcl_tensor *input)
{
	mcl_tensor_reset (lay -> output);
	mcl_tensor_mul (lay -> weights, input, lay -> output);
	mcl_tensor_add (lay -> output, lay -> biases);
	lay -> activation -> function (lay -> output);
}

void mcl_layer_backward (mcl_layer *lay, mcl_tensor *input, mcl_tensor *output_grad)
{
	lay -> activation -> function_d (lay -> output, lay -> delta);
	mcl_tensor_mul_elem (lay -> delta, lay -> output_grad);
	mcl_tensor_add (lay -> bias_grad, lay -> delta);
	mcl_tensor_mul_tr (lay -> delta, input, lay -> weight_grad);
	if (output_grad) {
		mcl_tensor_reset (output_grad);
		mcl_tensor_mul_tl (lay -> weights, lay -> delta, output_grad);
	}
}

void mcl_layer_scale_grad (mcl_layer *lay, float scalar)
{
	mcl_tensor_scale (lay -> weight_grad, scalar);
	mcl_tensor_scale (lay -> bias_grad, scalar);
}

void mcl_layer_apply_grad (mcl_layer *lay)
{
	mcl_tensor_add (lay -> weights, lay -> weight_grad);
	mcl_tensor_add (lay -> biases, lay -> bias_grad);
}

void mcl_layer_delete (mcl_layer *lay)
{
	if (lay == NULL)
		return;

	mcl_tensor_delete (lay -> weights);
	mcl_tensor_delete (lay -> biases);
	mcl_tensor_delete (lay -> output);
	mcl_tensor_delete (lay -> weight_grad);
	mcl_tensor_delete (lay -> bias_grad);
	mcl_tensor_delete (lay -> delta);
	mcl_tensor_delete (lay -> output_grad);
	free (lay);
	lay = NULL;
}
