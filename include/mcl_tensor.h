#ifndef MCL_TENSOR_H
#define MCL_TENSOR_H

#include <stddef.h>

typedef struct mcl_tensor {
	float *ten;
	int row;
	int col;
} mcl_tensor;

mcl_tensor* mcl_tensor_create (int row, int col);

size_t mcl_tensor_size (mcl_tensor *ten);		// returns size in bytes

void mcl_tensor_reset (mcl_tensor *ten);	// sets all elements to 0

void mcl_tensor_print (mcl_tensor *ten);

void mcl_tensor_random_uniform (mcl_tensor *ten);	// randomizes from U(0, 1)

void mcl_tensor_random_normal (mcl_tensor *ten);	// randomizes from N(0, 1)

void mcl_tensor_add_scalar (mcl_tensor *ten, float scalar);

void mcl_tensor_scale (mcl_tensor *ten, float scalar);

void mcl_tensor_transpose (mcl_tensor *ten, mcl_tensor *ten_t);

void mcl_tensor_add (mcl_tensor *ten_a, mcl_tensor *ten_b);

void mcl_tensor_mul (mcl_tensor *left, mcl_tensor *right, mcl_tensor *res);

void mcl_tensor_mul_tl (mcl_tensor *left, mcl_tensor *right, mcl_tensor *res);

void mcl_tensor_mul_tr (mcl_tensor *left, mcl_tensor *right, mcl_tensor *res);

void mcl_tensor_dropout (mcl_tensor *ten, float dropout);

void mcl_tensor_multiply (mcl_tensor *ten_l, mcl_tensor *ten_r, mcl_tensor *res);

void mcl_tensor_add_multiply (mcl_tensor *ten_l, mcl_tensor *ten_r, mcl_tensor *res);

void mcl_tensor_delete (mcl_tensor *ten);

#endif
