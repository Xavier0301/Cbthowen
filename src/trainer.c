#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "unistd.h"

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "data_manager.h"

void train(u32 filter_inputs, u32 filter_entries, u32 filter_hashes, u32 bits_per_input, u32 dim1_block_size, u32 dim2_block_size, u32 bleach_max, u32 saving_option) {
    double train_val_ratio = 0.9;

    printf("*TRAINING*\n");
    model_t model;

    model_init_params_t params = {
        .num_classes = 10,

        .num_inputs = MNIST_IM_SIZE,
        .bits_per_input = bits_per_input,

        .dim1_block_size = dim1_block_size,
        .dim2_block_size = dim2_block_size,

        .filter_hashes = filter_hashes,
        .filter_inputs = filter_inputs,
        .filter_entries = filter_entries,
    };

    model_init(&model, &params);

    u32 num_test = MNIST_NUM_TEST;
    printf("Loading test dataset (%u)\n", num_test);
    mat_u8 test_data;
    matrix_u8_init(&test_data, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_data, test_labels, num_test);

    u32 num_total = MNIST_NUM_TRAIN;
    u32 num_train = num_total * train_val_ratio;
    u32 num_val = num_total - num_train;

    printf("Loading total train dataset (%u)\n", num_total);
    mat_u8 tmp_train_patterns;
    matrix_u8_init(&tmp_train_patterns, num_total, MNIST_IM_SIZE);
    unsigned char* tmp_train_labels = calloc(num_total, sizeof(*tmp_train_labels));
    load_infimnist(tmp_train_patterns, tmp_train_labels, num_total);

    printf("Binarizing train dataset\n");
    char* reordering;
    char* encoding;
#if defined(STRIDED_ENCODING) && defined(REORDER_FIRST)
    reordering = "REORDER_FIRST";
    encoding = "STRIDED_ENCODING";
#elif defined(STRIDED_ENCODING) && !defined(REORDER_FIRST)
    reordering = "REORDER_LAST";
    encoding = "STRIDED_ENCODING";
#elif defined(LOCAL_STRIDED_ENCODING) && defined(REORDER_FIRST)
    reordering = "REORDER_FIRST";
    encoding = "LOCAL_STRIDED_ENCODING";
#elif defined(LOCAL_STRIDED_ENCODING) && !defined(REORDER_FIRST)
    reordering = "REORDER_LAST";
    encoding = "LOCAL_STRIDED_ENCODING";
#elif defined(LOCAL_ENCODING) && defined(REORDER_FIRST)
    reordering = "REORDER_FIRST";
    encoding = "LOCAL_ENCODING";
#elif defined(LOCAL_ENCODING) && !defined(REORDER_FIRST)
    reordering = "REORDER_LAST";
    encoding = "LOCAL_ENCODING";
#endif
    char filename[200];
    sprintf(
        filename, 
        "thresholds_%u_%u_%s_%s.dat", 
        model.p.num_inputs, model.p.bits_per_input, 
        reordering, encoding
    );

    mat_u8 tmp_thresh;
    matrix_u8_init(&tmp_thresh, model.p.num_inputs, model.p.bits_per_input);

    if(access(filename, F_OK) == 0) {
        FILE* f = fopen(filename, "r");
        read_matrix_u8(
            f, tmp_thresh, model.p.num_inputs_encoded
        );
#ifdef REORDER_FIRST
        reorder_thresholds(
            model.encoding_thresholds, tmp_thresh, 
            model.input_order, 
            model.p.bits_per_input, model.p.num_inputs
        );
#else 
        model.encoding_thresholds = tmp_thresh;
#endif
    } else {
        set_thresholds(
            &tmp_thresh,
            tmp_train_patterns, 
            MNIST_IM_SIZE, num_train, bits_per_input
        );
        FILE* f = fopen(filename, "w");
        write_matrix_u8(
            f, tmp_thresh, model.p.num_inputs_encoded
        );
#ifdef REORDER_FIRST
        reorder_thresholds(
            model.encoding_thresholds, tmp_thresh, 
            model.input_order, 
            model.p.bits_per_input, model.p.num_inputs
        );
#else 
        model.encoding_thresholds = tmp_thresh;
#endif
    }

    mat_u8 training_data = { 
        .stride = tmp_train_patterns.stride, 
        .data = MATRIX_AXIS1(tmp_train_patterns, 0)
    };
    mat_u8 validation_data = { 
        .stride = tmp_train_patterns.stride, 
        .data = MATRIX_AXIS1(tmp_train_patterns, num_train)
    };

    unsigned char* train_labels = tmp_train_labels;
    unsigned char* val_labels = tmp_train_labels + num_train;

    // print_binarized_image_raw(binarized_infimnist, infimnist_labels, 0, bits_per_input);
    // print_binarized_image_raw(binarized_val, val_labels, 0, bits_per_input);

    printf("Training\n");

    for(u32 sample_it = 0; sample_it < num_train; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(training_data, sample_it), train_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %u\n", sample_it);
    }

    u64 max_entry = 0;
    for(u32 discr_it = 0; discr_it < model.p.num_classes; ++discr_it) {
        for(u32 filter_it = 0; filter_it < model.p.num_filters; ++filter_it) {
            for(u32 entry_it = 0; entry_it < model.p.filter_entries; ++entry_it) {
                u64 el = *TENSOR3D(model.filters, discr_it, filter_it, entry_it);
                if(el > max_entry)
                    max_entry = el;
            }
        }
    }
    printf("Max entry: %llu\n", max_entry);

    printf("Testing to find optimal bleaching threshold\n");

    u32 best_bleach = 0;
    double best_accuracy = 0;
    for(u32 bleach = 1; bleach < bleach_max; bleach+=1) {

        model.p.bleach = bleach;

        u32 correct = 0;
        for(u32 sample_it = 0; sample_it < num_val; ++sample_it) {
            u32 class = model_predict2(&model, MATRIX_AXIS1(validation_data, sample_it));
            correct += (class == val_labels[sample_it]);
        }

        double accuracy = ((double) correct) / ((double) num_val);
        printf("Bleach %u. Accuracy %u/%u (%f%%)\n", bleach, correct, num_val, 100 * accuracy);

        if(accuracy >= best_accuracy) {
            best_bleach = bleach;
            best_accuracy = accuracy;
        }
    }

    model.p.bleach = best_bleach;
    printf("Best bleach: %u (%lf)\n", best_bleach, best_accuracy);

    printf("Accuracy on test set\n");
    u32 correct = 0;
    for(u32 sample_it = 0; sample_it < num_test; ++sample_it) {
        u32 class = model_predict2(&model, MATRIX_AXIS1(test_data, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_test);
    printf("test_accuracy, %u, %u, %.2f, %u\n", correct, num_test, 100 * accuracy, best_bleach);

    if(saving_option == 2) {
        pmodel_t pmodel;

        model_bleach(&model);
        model_pack(&model, &pmodel);

        // accuracy of packed model
        correct = 0;
        for(u32 sample_it = 0; sample_it < num_test; ++sample_it) {
            u32 class = pmodel_predict(&pmodel, MATRIX_AXIS1(test_data, sample_it));
            correct += (class == test_labels[sample_it]);
        }
        accuracy = ((double) correct) / ((double) num_test);
        printf("packed_test_accuracy, %u, %u, %.2f, %u\n", correct, num_test, 100 * accuracy, best_bleach);

        char model_name[256];
        sprintf(model_name, "pmodel_%u_%u_%u_%u_%u_%u_%u.dat", filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, bleach_max);

        printf("Saving model %s\n", model_name);

        write_pmodel(model_name, &pmodel);
    } else if(saving_option == 1) {
        char model_name[256];
        sprintf(model_name, "model_%u_%u_%u_%u_%u_%u_%u.dat", filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, bleach_max);

        printf("Saving model %s\n", model_name);

        write_model(model_name, &model);
    }


}

int main(int argc, char *argv[]) {                              
    if(argc < 9) {
        printf("Error: usage: %s filter_inputs filter_entries filter_hashes bits_per_input dim1_block_size dim2_block_size bleach_max saving_option\n", argv[0]);
        printf("\tsaving_options: 0 for nothing, 1 for model, 2 for bleach/packed model\n");
        printf("\tExample usage: %s 28 1024 2 2 14 28 10 1\n", argv[0]);

        return 1;
    }

    u32 filter_inputs = atoi(argv[1]);
    u32 filter_entries = atoi(argv[2]);
    u32 filter_hashes = atoi(argv[3]);
    u32 bits_per_input = atoi(argv[4]);
    u32 dim1_block_size = atoi(argv[5]);  
    u32 dim2_block_size = atoi(argv[6]);
    u32 bleach_max = atoi(argv[7]);
    u32 saving_option = atoi(argv[8]);

    printf("Training with parameters filter_inputs=%u, filter_entries=%u, filter_hashes=%u, bits_per_input=%u, dim1_block_size=%u, dim2_block_size=%u, bleach=..%u, saving_option=%u\n", 
                                    filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, bleach_max, saving_option);

    train(filter_inputs, filter_entries, filter_hashes, bits_per_input, dim1_block_size, dim2_block_size, bleach_max, saving_option);
}
