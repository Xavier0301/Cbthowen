#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "data_manager.h"

void train(size_t filter_inputs, size_t filter_entries, size_t filter_hashes, size_t bits_per_input, size_t block_size_div, size_t bleach_max, size_t saving_option) {
    double train_val_ratio = 0.9;

    printf("*TRAINING*\n");
    model_t model;

    model_init_params_t params = {
        .num_classes = 10,

        .num_inputs = MNIST_IM_SIZE * bits_per_input,
        .bits_per_input = bits_per_input,

        .block_size_div = block_size_div,

        .filter_hashes = filter_hashes,
        .filter_inputs = filter_inputs,
        .filter_entries = filter_entries,
    };

    model_init(&model, &params);

    size_t num_test = MNIST_NUM_TEST;
    printf("Loading test dataset (%zu)\n", num_test);
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    size_t num_total = MNIST_NUM_TRAIN;
    size_t num_train = num_total * train_val_ratio;
    size_t num_val = num_total - num_train;

    printf("Loading total train dataset (%zu)\n", num_total);
    mat_u8 tmp_train_patterns;
    matrix_u8_init(&tmp_train_patterns, num_total, MNIST_IM_SIZE);
    unsigned char* tmp_train_labels = calloc(num_total, sizeof(*tmp_train_labels));
    load_infimnist(tmp_train_patterns, tmp_train_labels, num_total);

    printf("Binarizing train dataset\n");
    double mean[MNIST_IM_SIZE];
    double variance[MNIST_IM_SIZE];

    mat_u8_mean(mean, tmp_train_patterns, MNIST_IM_SIZE, num_total);
    mat_u8_variance(variance, tmp_train_patterns, MNIST_IM_SIZE, num_total, mean);

    mat_u8 binarized_tmp_train;
    matrix_u8_init(&binarized_tmp_train, num_total, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix_meanvar(
        binarized_tmp_train, 
        tmp_train_patterns, 
        mean, variance,
        num_total, bits_per_input);

    mat_u8 binarized_train = { 
        .stride = binarized_tmp_train.stride, 
        .data = binarized_tmp_train.data
    };
    mat_u8 binarized_val = { 
        .stride = binarized_tmp_train.stride, 
        .data = binarized_tmp_train.data + num_train * binarized_tmp_train.stride
    };

    unsigned char* train_labels = tmp_train_labels;
    unsigned char* val_labels = tmp_train_labels + num_train;

    printf("Binarizing test dataset\n"); 
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix_meanvar(
        binarized_test, 
        test_patterns, 
        mean, variance,
        num_test, bits_per_input);


    // print_binarized_image(&binarized_test, test_labels, 0, 2);
    // print_binarized_image_raw(binarized_infimnist, infimnist_labels, 0, bits_per_input);
    // print_binarized_image_raw(binarized_val, val_labels, 0, bits_per_input);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < num_train; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_train, sample_it), train_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %zu\n", sample_it);
    }

    u64 max_entry = 0;
    for(size_t discr_it = 0; discr_it < model.p.num_classes; ++discr_it) {
        for(size_t filter_it = 0; filter_it < model.p.num_filters; ++filter_it) {
            for(size_t entry_it = 0; entry_it < model.p.filter_entries; ++entry_it) {
                u64 el = *TENSOR3D(model.filters, discr_it, filter_it, entry_it);
                if(el > max_entry)
                    max_entry = el;
            }
        }
    }
    printf("Max entry: %llu\n", max_entry);

    printf("Testing to find optimal bleaching threshold\n");

    size_t best_bleach = 0;
    double best_accuracy = 0;
    for(size_t bleach = 1; bleach < bleach_max; bleach+=1) {

        model.p.bleach = bleach;

        size_t correct = 0;
        for(size_t sample_it = 0; sample_it < num_val; ++sample_it) {
            size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_val, sample_it));
            correct += (class == val_labels[sample_it]);
        }

        double accuracy = ((double) correct) / ((double) num_val);
        printf("Bleach %zu. Accuracy %zu/%zu (%f%%)\n", bleach, correct, num_val, 100 * accuracy);

        if(accuracy >= best_accuracy) {
            best_bleach = bleach;
            best_accuracy = accuracy;
        }
    }

    model.p.bleach = best_bleach;
    printf("Best bleach: %zu (%lf)\n", best_bleach, best_accuracy);

    printf("Accuracy on test set\n");
    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < num_test; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_test);
    printf("test_accuracy, %zu, %zu, %.2f, %zu\n", correct, num_test, 100 * accuracy, model.p.block_size);

    if(saving_option == 2) {
        printf("Saving model\n");
        pmodel_t pmodel;

        model_bleach(&model);
        model_pack(&model, &pmodel);

        write_pmodel("pmodel.dat", &pmodel);
    } else if(saving_option == 1) {
        printf("Saving model\n");
        write_model("model.dat", &model);
    }


}

int main(int argc, char *argv[]) {                              
    if(argc < 8) {
        printf("Error: usage: %s filter_inputs filter_entries filter_hashes bits_per_input block_size_div (0 for max size) bleach_max saving_option\n", argv[0]);
        printf("\tSaving options: 0 for nothing, 1 for model, 2 for bleach/packed model\n");
        printf("\tExample usage: %s 28 1024 2 2 0 12 0\n", argv[0]);

        return 1;
    }

    size_t filter_inputs = atoi(argv[1]);
    size_t filter_entries = atoi(argv[2]);
    size_t filter_hashes = atoi(argv[3]);
    size_t bits_per_input = atoi(argv[4]);
    size_t block_size_div = atoi(argv[5]);
    size_t bleach_max = atoi(argv[6]);
    size_t saving_option = atoi(argv[7]);

    printf("Training with parameters filter_inputs=%zu, filter_entries=%zu, filter_hashes=%zu, bits_per_input=%zu, block_size_div=%zu, bleach=..%zu, saving_option=%zu\n", filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size_div, bleach_max, saving_option);

    train(filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size_div, bleach_max, saving_option);
}
