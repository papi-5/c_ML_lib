#include <stdlib.h>
#include "mcl_tensor.h"
#include "mcl_layer.h"

mcl_layer* mcl_layer_create (int row, int col)
{
	mcl_layer *lay = malloc (sizeof (mcl_layer));

	lay -> weights = mcl_tensor_create (row, col);
	lay -> weights_t = mcl_tensor_create (col, row);
	lay -> biases = mcl_tensor_create (1, col);
	lay -> output = mcl_tensor_create (1, col);
	lay -> output_t = mcl_tensor_create (col, 1);
	lay -> weight_grad = mcl_tensor_create (row, col);
	lay -> bias_grad = mcl_tensor_create (1, col);
	lay -> input_grad = mcl_tensor_create (1, col);
	lay -> cost_grad = mcl_tensor_create (1, col);

	return lay;
}

size_t mcl_layer_size (mcl_layer *lay)
{
	size_t size = mcl_tensor_size(lay -> weights);
	size += mcl_tensor_size(lay -> weights_t);
	size += mcl_tensor_size(lay -> biases);
	size += mcl_tensor_size(lay -> output);
	size += mcl_tensor_size(lay -> output_t);
	size += mcl_tensor_size(lay -> weight_grad);
	size += mcl_tensor_size(lay -> bias_grad);
	size += mcl_tensor_size(lay -> input_grad);
	size += mcl_tensor_size(lay -> cost_grad);
	size += sizeof (mcl_layer);

	return size;
}

void mcl_layer_weights_rand_uniform (mcl_layer *lay, float lower, float upper)
{
	mcl_tensor_random_uniform (lay -> weights);
	mcl_tensor_scale (lay -> weights, upper - lower);
	mcl_tensor_add_scalar (lay -> weights, lower);
	mcl_tensor_transpose (lay -> weights, lay -> weights_t);
}

void mcl_layer_weights_rand_normal (mcl_layer *lay, float mean, float stddev)
{
	mcl_tensor_random_normal (lay -> weights);
	mcl_tensor_scale (lay -> weights, stddev);
	mcl_tensor_add_scalar (lay -> weights, mean);
	mcl_tensor_transpose (lay -> weights, lay -> weights_t);
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

void mcl_layer_delete (mcl_layer *lay)
{
	if (lay == NULL)
		return;

	mcl_tensor_delete (lay -> weights);
	mcl_tensor_delete (lay -> weights_t);
	mcl_tensor_delete (lay -> biases);
	mcl_tensor_delete (lay -> output);
	mcl_tensor_delete (lay -> output_t);
	mcl_tensor_delete (lay -> weight_grad);
	mcl_tensor_delete (lay -> bias_grad);
	mcl_tensor_delete (lay -> input_grad);
	mcl_tensor_delete (lay -> cost_grad);
	free (lay);
	lay = NULL;
}
