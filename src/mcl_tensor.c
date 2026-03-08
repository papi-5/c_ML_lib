#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "mcl_tensor.h"

#define TILE_DIM 32

mcl_tensor* mcl_tensor_create (int row, int col)
{
	mcl_tensor *ten = malloc(sizeof (mcl_tensor));

	ten -> row = row;
	ten -> col = col;
	ten -> ten = aligned_alloc (64, (row * col) * sizeof (float));

	return ten;
}

size_t mcl_tensor_size (mcl_tensor *ten)
{
	size_t size = sizeof (float) * ten -> row * ten -> col;
	size += sizeof (mcl_tensor);

	return size;
}

void mcl_tensor_reset (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		ten -> ten[i] = 0;
}

void mcl_tensor_print (mcl_tensor *ten)
{
	int row = ten -> row;
	int col = ten -> col;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++)
			printf ("%f ", ten -> ten[i * col + j]);
		printf ("\n");
	}
	printf ("\n");
}

void mcl_tensor_random_uniform (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;

	srand (time (NULL));

	for (int i = 0; i < length; i++) {
		ten -> ten[i] = (float)rand() / RAND_MAX;
	}
}

void mcl_tensor_random_normal (mcl_tensor *ten)
{
	int length = ten -> row * ten -> col;
	float sum;

	srand (time (NULL));

	for (int i = 0; i < length; i++) {
		sum = 0;
		for (int j = 0; j < 12; j++)
			sum += (float)rand() / RAND_MAX;
		ten -> ten[i] = sum - 6;
	}
}

void mcl_tensor_add_scalar (mcl_tensor *ten, float scalar)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		ten -> ten[i] += scalar;
}

void mcl_tensor_scale (mcl_tensor *ten, float scalar)
{
	int length = ten -> row * ten -> col;

	for (int i = 0; i < length; i++)
		ten -> ten[i] *= scalar;
}

void mcl_tensor_transpose (mcl_tensor *ten, mcl_tensor *ten_t)
{
	int row = ten -> row;
	int col = ten -> col;
	int length = row * col;

	for (int i = 0; i < length; i++)
		ten_t -> ten[i] = ten -> ten[(i % row) * col + i / row];
}

void mcl_tensor_add (mcl_tensor *ten_a, mcl_tensor *ten_b)
{
	int length = ten_a -> row * ten_a -> col;

	for (int i = 0; i < length; i++)
		ten_a -> ten[i] += ten_b -> ten[i];
}

static void tile_mul (
	float *A, float *B, float *C,
	int m, int k, int n,
	int lda, int ldb, int ldc
)
{
	for (int i = 0; i < m; i++) {
		for (int l = 0; l < k; l++) {
			__m256 a_scalar = _mm256_set1_ps (A[i * lda + l]);
			for (int j = 0; j < n - 7; j += 8) {
				__m256 c_vec = _mm256_loadu_ps (&C[i * ldc + j]);
				__m256 b_vec = _mm256_loadu_ps (&B[l * ldb + j]);
				c_vec = _mm256_fmadd_ps (a_scalar, b_vec, c_vec);
				_mm256_storeu_ps (&C[i * ldc + j], c_vec);
			}
			for (int j = n - (n % 8); j < n; j++) {
				C[i * ldc + j] += A[i * lda + l] * B[l * ldb + j];
			}
		}
	}
}

static void tile_mul_t (
	float *A, float *B, float *C,
	int m, int k, int n,
	int lda, int ldb, int ldc
)
{
	for (int i = 0; i < m; i++) {
		for (int l = 0; l < k; l++) {
			__m256 a_scalar = _mm256_set1_ps (A[l * lda + i]);
			for (int j = 0; j < n - 7; j += 8) {
				__m256 c_vec = _mm256_loadu_ps (&C[i * ldc + j]);
				__m256 b_vec = _mm256_loadu_ps (&B[l * ldb + j]);
				c_vec = _mm256_fmadd_ps (a_scalar, b_vec, c_vec);
				_mm256_storeu_ps (&C[i * ldc + j], c_vec);
			}
			for (int j = n - (n % 8); j < n; j++) {
				C[i * ldc + j] += A[l * lda + i] * B[l * ldb + j];
			}
		}
	}
}

void mcl_tensor_mul (mcl_tensor *left, mcl_tensor *right, mcl_tensor *res)
{
	int rows = left -> row;
	int kdim = left -> col;
	int cols = right -> col;
	for (int i = 0; i < rows; i += TILE_DIM) {
		for (int j = 0; j < cols; j += TILE_DIM) {
			for (int l = 0; l < kdim; l += TILE_DIM) {
				int m = rows - i < TILE_DIM ? (rows - i) : TILE_DIM;
				int k = kdim - l < TILE_DIM ? (kdim - l) : TILE_DIM;
				int n = cols - j < TILE_DIM ? (cols - j) : TILE_DIM;
				tile_mul (
					&(left -> ten[i * kdim + l]), &(right ->ten[l * cols + j]), &(res -> ten[i * cols + j]),
					m, k, n,
					kdim, cols, cols
				);
			}
		}
	}
}

void mcl_tensor_mul_t (mcl_tensor *left, mcl_tensor *right, mcl_tensor *res)
{
	int col_l = left -> col;
	int kdim = left -> row;
	int col_r = right -> col;
	for (int i = 0; i < col_l; i += TILE_DIM) {
		for (int j = 0; j < col_r; j += TILE_DIM) {
			for (int l = 0; l < kdim; l += TILE_DIM) {
				int m = col_l - i < TILE_DIM ? (col_l - i) : TILE_DIM;
				int k = kdim - l < TILE_DIM ? (kdim - l) : TILE_DIM;
				int n = col_r - j < TILE_DIM ? (col_r - j) : TILE_DIM;
				tile_mul_t (
					&(left -> ten[l * col_l + i]), &(right ->ten[l * col_r + j]), &(res -> ten[i * col_r + j]),
					m, k, n,
					col_l, col_r, col_r
				);
			}
		}
	}
}

static float tensor_dot (mcl_tensor *ten_l, mcl_tensor *ten_r, int row, int col)
{
	int lColl = ten_l -> col;
	int rColl = ten_r -> col;
	float sum = 0;

	for (int i = 0; i < lColl; i++)
		sum += ten_l -> ten[row * lColl + i] * ten_r -> ten[i * rColl + col];

	return sum;
}

void mcl_tensor_multiply (mcl_tensor *ten_l, mcl_tensor *ten_r, mcl_tensor *res)
{
	int lRows = ten_l -> row;
	int rColl = ten_r -> col;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++)
			res -> ten[i * rColl + j] = tensor_dot (ten_l, ten_r, i, j);
	}
}

void mcl_tensor_add_multiply (mcl_tensor *ten_l, mcl_tensor *ten_r, mcl_tensor *res)
{
	int lRows = ten_l -> row;
	int rColl = ten_r -> col;

	for (int i = 0; i < lRows; i++) {
		for (int j = 0; j < rColl; j++)
			res -> ten[i * rColl + j] += tensor_dot (ten_l, ten_r, i, j);
	}
}

void mcl_tensor_delete (mcl_tensor *ten)
{
	if (ten == NULL)
		return;

	free (ten -> ten);
	free (ten);
	ten = NULL;
}
