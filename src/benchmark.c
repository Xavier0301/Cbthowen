#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "data_manager.h"

void inference(model_t* model, u32 num_samples, u32 num_models) {
    u8* input = calloc(model->p.num_inputs, sizeof(*input));
    for(u32 model_it = 0; model_it < num_models; ++model_it) {
        for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
            model_predict2(model, input);
        }
    }

    printf("[Benchmarking] DONE\n");
}

void training(model_t* model, u32 num_samples, u32 num_models) {
    u8* input = calloc(model->p.num_inputs, sizeof(*input));
    u64 target = 0;
    for(u32 model_it = 0; model_it < num_models; ++model_it) {
        for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
            model_train(model, input, target);
        }
    }

    printf("[Benchmarking] DONE\n");
}

void packed_inference(pmodel_t* model, u32 num_samples, u32 num_models) {
    u8* input = calloc(model->p.num_inputs, sizeof(*input));
    for(u32 model_it = 0; model_it < num_models; ++model_it) {
        for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
            pmodel_predict(model, input);
        }
    }

    printf("[Benchmarking] DONE\n");
}

int main(int argc, char *argv[]) {                              
    if(argc < 5) {
        printf("Error: usage: %s model_size type mode num_samples\n", argv[0]);
        printf("\tModel size: s for small, m for medium, l for large\n");
        printf("\tModel Type: n for normal, p for packed\n");
        printf("\tMode: i for inference, t for training\n");
        printf("\tNum models: number of models to run\n");
        printf("\tExample usage: %s small normal inference 1000 1\n", argv[0]);
        printf("\tThis is intended to be used alongside a profiler like sample \n\t\tto get accurate breakdown of operations during inference and training.\n\tOr use time <command>!!!\n");

        return 1;
    }

    char* model_size = argv[1];
    char* model_type = argv[2];
    char* mode = argv[3];
    u32 num_samples = atoi(argv[4]);
    u32 num_models = atoi(argv[5]);

    printf("Benchmarking %s (of type %s) with mode %s and %u samples [%u models]\n", model_size, model_type, mode, num_samples, num_models);

    model_init_params_t params;
    if(model_size[0] == 's') {
        params = (model_init_params_t) {
            .num_classes = 10,

            .num_inputs = MNIST_IM_SIZE,
            .bits_per_input = 2,

            .dim1_block_size = 392,
            .dim2_block_size = 1,

            .filter_hashes = 2,
            .filter_inputs = 28,
            .filter_entries = 1024,
        };
    } else if(model_size[0] == 'm') {
        params = (model_init_params_t) {
            .num_classes = 10,

            .num_inputs = MNIST_IM_SIZE,
            .bits_per_input = 3,

            .dim1_block_size = 392,
            .dim2_block_size = 1,

            .filter_hashes = 2,
            .filter_inputs = 28,
            .filter_entries = 2048,
        };
    } else if(model_size[0] == 'l') {
        params = (model_init_params_t) {
            .num_classes = 10,

            .num_inputs = MNIST_IM_SIZE,
            .bits_per_input = 6,

            .dim1_block_size = 392,
            .dim2_block_size = 1,

            .filter_hashes = 4,
            .filter_inputs = 49,
            .filter_entries = 8192,
        };
    } else {
        printf("Error: invalid model size %s\n", model_size);
        exit(1);
    }

    model_t model;
    model_init(&model, &params);

    if(model_type[0] == 'n') {
        if(mode[0] == 'i') {
            inference(&model, num_samples, num_models);
        } else if(mode[0] == 't') {
            training(&model, num_samples, num_models);
        } else {
            printf("Error: invalid mode %s\n", mode);
        }
    } else if(model_type[0] == 'p') {
        pmodel_t pmodel;
        model_pack(&model, &pmodel);

        if(mode[0] == 'i') {
            packed_inference(&pmodel, num_samples, num_models);
        } else {
            printf("Error: invalid mode %s for model type: packed\n", mode);
        }
    } else {
        printf("Error: invalid model type %s\n", model_type);
        exit(1);
    }
}
