#ifndef MCL_DATASET_H
#define MCL_DATASET_H

#include "mcl_tensor.h"

typedef enum {
    MCL_CLASSIFICATION,
    MCL_REGRESSION
} mcl_task_type;

typedef enum {
    MCL_FIRST,
    MCL_LAST
} mcl_label_position;

typedef struct mcl_dataset {
    mcl_tensor **train;
    mcl_tensor **test;
    int train_size;
    int test_size;
    int input_size;
    int output_size;
    mcl_task_type task;
    mcl_label_position label_position;
} mcl_dataset;

mcl_dataset* mcl_dataset_create (mcl_task_type task, mcl_label_position label_position, int input_size, int output_size);

void mcl_dataset_load_train (mcl_dataset *data, const char *path);

void mcl_dataset_load_test (mcl_dataset *data, const char *path);

void mcl_dataset_load_split (mcl_dataset *data, const char *path, float ratio);

void mcl_dataset_shuffle (mcl_tensor **data_points, int data_size);

int mcl_dataset_train_samples (mcl_dataset *data);

int mcl_dataset_test_samples (mcl_dataset *data);

void mcl_dataset_delete (mcl_dataset *data);

#endif