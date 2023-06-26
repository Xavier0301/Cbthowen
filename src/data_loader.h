#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "assert.h"

#include "math.h"
#include "distributions.h"

#include "tensor.h"
#include "model.h"

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

void load_mnist_file(u8_matrix_t patterns, uint8_t* labels, char* image_path, char* label_path, size_t num_samples, size_t offset);
void load_mnist_train(u8_matrix_t patterns, uint8_t* labels, size_t num_samples);
void load_mnist_test(u8_matrix_t patterns, uint8_t* labels, size_t num_samples);
void load_infimnist(u8_matrix_t patterns, uint8_t* labels, size_t num_samples);
void load_infimnist_labels(uint8_t* labels, size_t num_samples);
void load_mnist_train_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset);
void load_mnist_test_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset);
void load_infimnist_offset(u8_matrix_t patterns, uint8_t* labels, size_t num_samples, size_t offset);
void load_infimnist_labels_offset(uint8_t* labels, size_t num_samples, size_t offset);

void binarize_matrix(u8_matrix_t result, u8_matrix_t dataset, size_t sample_size, size_t num_samples, size_t num_bits);
void binarize_matrix_meanvar(u8_matrix_t result, u8_matrix_t dataset, double* mean, double* variance, size_t sample_size, size_t num_samples, size_t num_bits);

void reorder_dataset(u8_matrix_t result, u8_matrix_t dataset, uint16_t* order, size_t num_samples, size_t num_elements);

void print_binarized_image_raw(u8_matrix_t m, uint8_t* labels, size_t index, size_t num_bits);
void print_binarized_image(u8_matrix_t m, uint8_t* labels, size_t index, size_t num_bits);
void print_image_raw(u8_matrix_t m, uint8_t* labels, size_t index);
void print_image(u8_matrix_t m, uint8_t* labels, size_t index);

void fill_input_random(uint8_t* input, size_t input_length);

