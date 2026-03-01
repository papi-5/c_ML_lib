#ifndef MCL_DATASET_H
#define MCL_DATASET_H

#include "mcl_tensor.h"

typedef struct mcl_dataset {
    mcl_tensor **train;
    mcl_tensor **test;
    int train_size;
    int test_size;
    int input_size;
    int output_size;
    int label_position;
} mcl_dataset;

mcl_dataset* mcl_dataset_create (int input_size, int output_size, int label_position);

void mcl_dataset_load_train (mcl_dataset *data, const char *path);

void mcl_dataset_load_test (mcl_dataset *data, const char *path);

void mcl_dataset_load_split (mcl_dataset *data, const char *path, float ratio);

void mcl_dataset_shuffle (mcl_tensor **data_points, int data_size);

void mcl_dataset_delete (mcl_dataset *data);

#endif