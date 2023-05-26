#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "helper.h"

void read_model(const char* filename, model_t* model);
void read_dataset(const char* filename, bmatrix_t* dataset, size_t* num_samples, size_t* sample_size);
void read_dataset_partial(const char* filename, bmatrix_t* dataset, size_t num_samples_to_fetch, size_t* num_samples_total, size_t* sample_size);

void read_matrix(FILE* f, matrix_t* matrix, size_t size);
void read_bmatrix(FILE* f, bmatrix_t* matrix, size_t size);
void read_tensor(FILE* f, tensor3d_t* tensor, size_t size);

void write_model(const char* filename, model_t* model);
void write_dataset(const char* filename, bmatrix_t* dataset, size_t num_samples, size_t sample_size);

void write_matrix(FILE* f, matrix_t* matrix, size_t size);
void write_bmatrix(FILE* f, bmatrix_t* matrix, size_t size);
void write_tensor(FILE* f, tensor3d_t* tensor, size_t size);
