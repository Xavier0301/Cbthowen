#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "assert.h"

#include "math.h"

#include "types.h"

#include "tensor.h"
#include "model.h"
#include "distributions.h"


// set appropriate path for data
#define MNIST_TRAIN_IMAGE "./data/train-images-idx3-ubyte"
#define MNIST_TRAIN_LABEL "./data/train-labels-idx1-ubyte"
#define MNIST_TEST_IMAGE "./data/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL "./data/t10k-labels-idx1-ubyte"

#define INFIMNIST_PATTERNS "./data/mnist8m-patterns-idx3-ubyte"
#define INFIMNIST_LABELS "./data/mnist8m-labels-idx1-ubyte"

#define MNIST_IM_SIZE 784 // 28*28
#define MNIST_SIDE_LEN 28
#define INFIMNIST_NUM_SAMPLES 8000000
#define MNIST_NUM_TRAIN 60000
#define MNIST_NUM_TEST 10000
#define MNIST_LEN_INFO_IMAGE 4
#define MNIST_LEN_INFO_LABEL 2

void load_mnist_file(mat_u8 patterns, u8* labels, char* image_path, char* label_path, size_t num_samples, size_t offset);
void load_mnist_train(mat_u8 patterns, u8* labels, size_t num_samples);
void load_mnist_test(mat_u8 patterns, u8* labels, size_t num_samples);
void load_infimnist(mat_u8 patterns, u8* labels, size_t num_samples);
void load_infimnist_labels(u8* labels, size_t num_samples);
void load_mnist_train_offset(mat_u8 patterns, u8* labels, size_t num_samples, size_t offset);
void load_mnist_test_offset(mat_u8 patterns, u8* labels, size_t num_samples, size_t offset);
void load_infimnist_offset(mat_u8 patterns, u8* labels, size_t num_samples, size_t offset);
void load_infimnist_labels_offset(u8* labels, size_t num_samples, size_t offset);

void binarize_matrix(mat_u8 result, mat_u8 dataset, size_t sample_size, size_t num_samples, size_t num_bits);
void binarize_matrix_meanvar(mat_u8 result, mat_u8 dataset, double* mean, double* variance, size_t num_samples, size_t num_bits);

void reorder_dataset(mat_u8 result, mat_u8 dataset, u16* order, size_t num_samples, size_t num_elements);

void print_binarized_image_raw(mat_u8 m, u8* labels, size_t index, size_t num_bits);
void print_binarized_image(mat_u8 m, u8* labels, size_t index, size_t num_bits);
void print_image_raw(mat_u8 m, u8* labels, size_t index);
void print_image(mat_u8 m, u8* labels, size_t index);

void fill_input_random(u8* input, size_t input_length);

