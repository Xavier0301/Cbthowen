#ifndef PACKED_MODEL_H
#define PACKED_MODEL_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>

#include "types.h"

#include "tensor.h"
#include "model.h"
#include "packing.h"

typedef struct pmodel_t_ {
    model_params_t p;

    u16* input_order; // of shape (#Inputs) with elements in [0; num_inputs_total)
    mat_u16 hash_parameters; // of shape (#Hashes, #Inputs)
    tensor_u8 filters; // packed, of shape (#Discriminators, #Filters, #Entries)
} pmodel_t;

void pmodel_init_buffers(pmodel_t* model);

// Assumes that source is bleached
void model_pack(model_t* source, pmodel_t* dest);

u64 pmodel_predict(pmodel_t* model, u8* input);

u64 pmodel_predict_backend(pmodel_t* model, mat_u16 hashes_buffer);

#endif // PACKED_MODEL_H
