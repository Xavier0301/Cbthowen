#include "model.h"

#include "time.h"

u8* reorder_buffer;
mat_u16 hashes_buffer; // used in predict2

void reorder_array(u8* result, u8* input, u16* order, u32 len) {
    for(u32 it = 0; it < len; ++it)
        result[it] = input[order[it]];
}

void randomize_input_order(u16* input_order, u32 len, u32 block_size) {
    for(u32 it = 0; it < len; ++it) {
        input_order[it] = it;
    }
    
    for(u32 it = 0; it < len; it += block_size) {
        u32 dyn_len = (it + block_size > len) ? len - it : block_size;
        shuffle_array(input_order + it, dyn_len);
    }
}

void generate_h3_values(mat_u16 values, u32 num_hashes, u32 num_inputs, u32 num_entries) {
    for(u32 i = 0; i < num_hashes; ++i) {
        for(u32 j = 0; j < num_inputs; ++j) {
            *MATRIX(values, i, j) = unif_rand(num_entries - 1);
        }
    }
}

void model_init(model_t* model, model_init_params_t* params) {
    // Params
    model->p.num_classes = params->num_classes;
    model->p.num_filters = params->num_inputs / params->filter_inputs;

    model->p.pad_zeros = (((params->num_inputs / params->filter_inputs) * params->filter_inputs) - params->num_inputs) % params->filter_inputs;
    model->p.num_inputs_total = params->num_inputs + model->p.pad_zeros;
    model->p.bits_per_input = params->bits_per_input;

    model->p.block_size = model->p.num_inputs_total / params->block_size_div;

    model->p.filter_hashes = params->filter_hashes;
    model->p.filter_inputs = params->filter_inputs;
    model->p.filter_entries = params->filter_entries;

    model->p.bleach = 1;

    // Buffer allocs
    model_init_buffers(model);

    // Buffer filling
    srand(time(NULL));
    randomize_input_order(model->input_order, model->p.num_inputs_total, model->p.block_size);

    generate_h3_values(model->hash_parameters, model->p.filter_hashes, model->p.filter_inputs, model->p.filter_entries);

}

void model_init_buffers(model_t* model){
    // Order
    model->input_order = calloc(model->p.num_inputs_total, sizeof(*model->input_order));

    // Hashes
    matrix_u16_init(&model->hash_parameters, model->p.filter_hashes, model->p.filter_inputs);

    // Filters
    tensor_u16_init(&model->filters, model->p.num_classes, model->p.num_filters, model->p.filter_entries);

    // Pre-alloc buffers used at inference time
    reorder_buffer = calloc(model->p.num_inputs_total, sizeof(*reorder_buffer));
    matrix_u16_init(&hashes_buffer, model->p.num_filters, model->p.filter_hashes);
}

// assumes input is already zero padded
u64 model_predict(model_t* model, u8* input) {
    reorder_array(reorder_buffer, input, model->input_order, model->p.num_inputs_total);

    u64 response_index = 0;
    u64 max_response = 0;
    for(u32 it = 0; it < model->p.num_classes; ++it) {
        u64 resp = discriminator_predict(model, it, reorder_buffer);
        if(resp >= max_response) {
            max_response = resp;
            response_index = it;
        }
    }

    return response_index;
}

void model_train(model_t* model, u8* input, u64 target) {    
    reorder_array(reorder_buffer, input, model->input_order, model->p.num_inputs_total);

    discriminator_train(model, target, reorder_buffer);
}

u64 discriminator_predict(model_t* model, u32 discriminator_index, u8* input) {
    u64 response = 0;
    u8* chunk = input;

    for(u32 it = 0; it < model->p.num_filters; ++it) {
        response += filter_check_membership(model, discriminator_index, it, chunk);
        chunk += model->p.filter_inputs;
    }

    return response;
}

void discriminator_train(model_t* model, u32 discriminator_index, u8* input) {
    u8* chunk = input;

    for(u32 it = 0; it < model->p.num_filters; ++it) {
        filter_add_member(model, discriminator_index, it, chunk);
        chunk += model->p.filter_inputs;
    }
}

/**
 * @brief 
 * 
 * @param result Resulting hash
 * @param input Boolean vector of shape (#inputs)
 * @param parameters Vector of shape (#inputs)
 */
u16 h3_hash(u8* input, u16* parameters, u32 num_inputs) {
    u16 result = parameters[0] * input[0];
    for(u32 j = 1; j < num_inputs; ++j) {
        result ^= parameters[j] * input[j];
    }

    return result;
}

// Can be replaced by an AND reduction (ONLY WHEN BLEACH=1)
int filter_check_membership(model_t* model, u32 discriminator_index, u32 filter_index, u8* input) {
    u16 hash_result;
    u16 entry;

    u16 minimum = 0xffff;
    for(u32 it = 0; it < model->p.filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->p.filter_inputs);
        entry = *TENSOR3D(model->filters, discriminator_index, filter_index, hash_result);
        if(entry <= minimum) minimum = entry;
    }

    return minimum >= model->p.bleach;
}

void filter_add_member(model_t* model, u32 discriminator_index, u32 filter_index, u8* input) {
    u16 hash_result;
    u16 entry;

    // Get minimum of all filter hash response
    u16 minimum = 0xffff;
    for(u32 it = 0; it < model->p.filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->p.filter_inputs);
        entry = *TENSOR3D(model->filters, discriminator_index, filter_index, hash_result);
        if(entry < minimum) minimum = entry;
    }

    // Increment the value of all minimum entries
    for(u32 it = 0; it < model->p.filter_hashes; ++it) {
        hash_result = h3_hash(input, MATRIX_AXIS1(model->hash_parameters, it), model->p.filter_inputs);
        entry = *TENSOR3D(model->filters, discriminator_index, filter_index, hash_result);
        if(entry == minimum) 
            *TENSOR3D(model->filters, discriminator_index, filter_index, hash_result) = minimum + 1;
    }
}

u16 filter_reduction(u16* filter, u16* hashes, u32 filter_hashes) {
    u16 min = 0xffff;
    for(u32 it = 0; it < filter_hashes; ++it) {
        u16 entry = filter[hashes[it]];
        if(entry < min) min = entry;
    }

    return min;
}

void perform_hashing(mat_u16 resulting_hashes, model_params_t* model_params, mat_u16 hash_parameters, u8* input) {
    u8* chunk = input;
    for(u32 chunk_it = 0; chunk_it < model_params->num_filters; ++chunk_it) {
        for(u32 hash_it = 0; hash_it < model_params->filter_hashes; ++hash_it) {
            *MATRIX(resulting_hashes, chunk_it, hash_it) = h3_hash(chunk, MATRIX_AXIS1(hash_parameters, hash_it), model_params->filter_inputs);
        }
        chunk += model_params->filter_inputs;
    }
}

u64 model_predict2(model_t* model, u8* input) {
    // Reorder
    reorder_array(reorder_buffer, input, model->input_order, model->p.num_inputs_total);

    // Hash
    perform_hashing(hashes_buffer, &model->p, model->hash_parameters, reorder_buffer);

    return model_predict_backend(model, hashes_buffer);
}

u64 model_predict_backend(model_t* model, mat_u16 hashes_buffer) {
    // Calculate popcounts for each discriminators
    u16 popcounts[model->p.num_classes];
    for(u32 discr_it = 0; discr_it < model->p.num_classes; ++discr_it)
        popcounts[discr_it] = 0;

    for(u32 filter_it = 0; filter_it < model->p.num_filters; ++filter_it) {
        for(u32 discr_it = 0; discr_it < model->p.num_classes; ++discr_it) {
            u16 resp = filter_reduction(TENSOR3D_AXIS2(model->filters, discr_it, filter_it), MATRIX_AXIS1(hashes_buffer, filter_it), model->p.filter_hashes);
            popcounts[discr_it] += (resp >= model->p.bleach);
        }
    }

    // for(u32 discr_it = 0; discr_it < model->p.num_classes; ++discr_it)
    //     printf("%d ", popcounts[discr_it]);
    // printf("\n");

    // Pick the argmax of popcounts
    u64 response_index = 0;
    u16 max_popcount = 0;
    for(u32 discr_it = 0; discr_it < model->p.num_classes; ++discr_it) {
        if(popcounts[discr_it] > max_popcount) {
            max_popcount = popcounts[discr_it];
            response_index = discr_it;
        }
    }

    return response_index;
}

void model_bleach(model_t* model) {
    for(u32 discr_it = 0; discr_it < model->p.num_classes; ++discr_it) {
        for(u32 filter_it = 0; filter_it < model->p.num_filters; ++filter_it) {
            for(u32 entry_it = 0; entry_it < model->p.filter_entries; ++entry_it) {
                u16* entry = TENSOR3D(model->filters, discr_it, filter_it, entry_it);
                *entry = (*entry >= model->p.bleach);
            }
        }
    }

    model->p.bleach = 1;
}

void print_model_params(model_params_t* model_params) {
    printf("Model parameters:\n");
    printf("\tTensor size: %u x %u x %u\n", model_params->num_classes, model_params->num_filters, model_params->filter_entries);
    printf("\tFilter params: %u (hashes) %u (inputs)\n", model_params->filter_hashes, model_params->filter_inputs);
    printf("\tInput size: %u (padding %u) (bits %u)\n", model_params->num_inputs_total, model_params->pad_zeros, model_params->bits_per_input);
    printf("\tOther: %u (block size) %u (bleach)\n", model_params->block_size, model_params->bleach);
}
