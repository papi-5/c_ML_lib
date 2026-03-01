#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mcl_dataset.h"
#include "mcl_tensor.h"

#define BUFFER_SIZE 8192

mcl_dataset* mcl_dataset_create (int input_size, int output_size, int label_position)
{
    mcl_dataset *data = malloc(sizeof (mcl_dataset));

    data -> input_size = input_size;
    data -> output_size = output_size;
    data -> label_position = label_position;

    return data;
}

static void one_hot_code_parse (mcl_tensor *ten, int output_size, int val)
{
    for (int i = 0; i < output_size; i++) {
        ten -> ten[i] = (i == val) ? 1 : 0;
    }
}

static void line_parse_csv (mcl_dataset *data, mcl_tensor **data_point, char *line)
{
    int input_size = data -> input_size;
    int output_size = data -> output_size;

    if (data -> label_position) {
        char *token = strtok (line, ",");
        for (int i = 0; i < input_size; i++) {
            float val = atof (token);
            data_point[0] -> ten[i] = val;
            token = strtok (NULL, ",");
        }
        int val = atoi (token);
        one_hot_code_parse (data_point[1], output_size, val);
    } else {
        char *token = strtok (line, ",");
        int val = atoi (token);
        one_hot_code_parse (data_point[1], output_size, val);
        for (int i = 0; i < input_size; i++) {
            token = strtok (NULL, ",");
            float val = atof (token);
            data_point[0] -> ten[i] = val;
        }
    }
}

void mcl_dataset_load_train (mcl_dataset *data, const char *path)
{
    FILE *file = fopen (path, "r");
    if (!file || !data) {
        printf ("Could not load train dataset.\n\n");
        return;
    }

    int num_lines = 0;
    char line[BUFFER_SIZE];
    while (fgets (line, sizeof (line), file)) {
        num_lines++;
    }
    rewind (file);

    int input_size = data -> input_size;
    int output_size = data -> output_size;
    data -> train_size = num_lines;
    data -> train = malloc (sizeof (mcl_tensor*) * num_lines * 2);
    for (int i = 0; i < num_lines; i++) {
        data -> train[i * 2] = mcl_tensor_create (input_size, 1);
        data -> train[i * 2 + 1] = mcl_tensor_create (output_size, 1);
    }

    int i = 0;
    while (fgets (line, sizeof (line), file)) {
        line_parse_csv (data, &(data -> train[i * 2]), line);
        i++;
    }
}

void mcl_dataset_load_test (mcl_dataset *data, const char *path)
{
    FILE *file = fopen (path, "r");
    if (!file || !data) {
        printf ("Could not load test dataset.\n\n");
        return;
    }

    int num_lines = 0;
    char line[BUFFER_SIZE];
    while (fgets (line, sizeof (line), file)) {
        num_lines++;
    }
    rewind (file);

    int input_size = data -> input_size;
    int output_size = data -> output_size;
    data -> test_size = num_lines;
    data -> test = malloc (sizeof (mcl_tensor*) * num_lines * 2);
    for (int i = 0; i < num_lines; i++) {
        data -> test[i * 2] = mcl_tensor_create (input_size, 1);
        data -> test[i * 2 + 1] = mcl_tensor_create (output_size, 1);
    }

    int i = 0;
    while (fgets (line, sizeof (line), file)) {
        line_parse_csv (data, &(data -> test[i * 2]), line);
        i++;
    }
}

void mcl_dataset_load_split (mcl_dataset *data, const char *path, float ratio)
{
    FILE *file = fopen (path, "r");
    if (!file || !data) {
        printf ("Could not load dataset.\n\n");
        return;
    }

    int num_lines = 0;
    char line[BUFFER_SIZE];
    while (fgets (line, sizeof (line), file)) {
        num_lines++;
    }
    rewind (file);

    int input_size = data -> input_size;
    int output_size = data -> output_size;
    mcl_tensor **data_points = malloc (sizeof (mcl_tensor*) * num_lines * 2);
    for (int i = 0; i < num_lines; i++) {
        data_points[i * 2] = mcl_tensor_create (input_size, 1);
        data_points[i * 2 + 1] = mcl_tensor_create (output_size, 1);
    }

    int i = 0;
    while (fgets (line, sizeof (line), file)) {
        line_parse_csv (data, &data_points[i * 2], line);
        i++;
    }

    mcl_dataset_shuffle (data_points, num_lines);

    data -> train_size = (int)(num_lines * ratio);
    data -> test_size = num_lines - data -> train_size;
    data -> train = malloc (sizeof (mcl_tensor*) * data -> train_size * 2);
    data -> test = malloc (sizeof (mcl_tensor*) * data -> test_size * 2);
    for (int i = 0; i < num_lines; i++) {
        if (i < data -> train_size) {
            data -> train[i * 2] = data_points[i * 2];
            data -> train[i * 2 + 1] = data_points[i * 2 + 1];
        } else {
            int test_i = i - data -> train_size;
            data -> test[test_i * 2] = data_points[i * 2];
            data -> test[test_i * 2 + 1] = data_points[i * 2 + 1];
        }
    }

    free (data_points);
}

void mcl_dataset_shuffle (mcl_tensor **data_points, int data_size)
{
    for (int i = 1; i < data_size; i++) {
        int ind = rand() % (i + 1);
        mcl_tensor *tmp = data_points[ind * 2];
        data_points[ind * 2] = data_points[i * 2];
        data_points[i * 2] = tmp;
        tmp = data_points[ind * 2 + 1];
        data_points[ind * 2 + 1] = data_points[i * 2 + 1];
        data_points[i * 2 + 1] = tmp;
    }
}

void mcl_dataset_delete (mcl_dataset *data)
{
    if (data == NULL)
        return;

    for (int i = 0; i < data -> train_size * 2; i++)
        free (data -> train[i]);
    for (int i = 0; i < data -> test_size * 2; i++)
        free (data -> test[i]);
    free (data -> train);
    free (data -> test);
    free (data);
    data = NULL;
}