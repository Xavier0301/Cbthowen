#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "data_manager.h"

void train(size_t filter_inputs, size_t filter_entries, size_t filter_hashes, size_t bits_per_input, size_t block_size) {
    double train_val_ratio = 0.9;

    printf("*TRAINING*\n");
    model_t model;

    size_t input_size = 784; 
    size_t num_inputs = input_size * bits_per_input;

    size_t num_classes = 10;

    model_init(&model, num_inputs, num_classes, filter_inputs, filter_entries, filter_hashes, bits_per_input, 1, block_size);

    size_t num_test = MNIST_NUM_TEST;
    printf("Loading test dataset (%zu)\n", num_test);
    u8_matrix_t test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    size_t num_train = MNIST_NUM_TRAIN * train_val_ratio;
    printf("Loading train dataset (%zu)\n", num_train);
    u8_matrix_t train_patterns;
    matrix_u8_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_infimnist(train_patterns, train_labels, num_train);

    size_t num_val = MNIST_NUM_TRAIN - num_train;
    printf("Loading val dataset (%zu)\n", num_val);
    u8_matrix_t val_patterns;
    matrix_u8_init(&val_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* val_labels = calloc(num_val, sizeof(*val_labels));
    load_infimnist_offset(val_patterns, val_labels, num_val, num_train);

    // print_image(val_patterns, val_labels, 0);
    // print_image(val_patterns, val_labels, 1);
    // print_image(val_patterns, val_labels, 2);

    printf("Binarizing test dataset\n");
    u8_matrix_t binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(binarized_test, test_patterns, MNIST_IM_SIZE, num_test, bits_per_input);

    printf("Binarizing train dataset\n");
    u8_matrix_t binarized_train;
    matrix_u8_init(&binarized_train, num_train, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(binarized_train, train_patterns, MNIST_IM_SIZE, num_train, bits_per_input); 

    printf("Binarizing val dataset\n");
    u8_matrix_t binarized_val;
    matrix_u8_init(&binarized_val, num_val, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(binarized_val, val_patterns, MNIST_IM_SIZE, num_val, bits_per_input);

    // print_binarized_image(&binarized_test, test_labels, 0, 2);
    // print_binarized_image_raw(binarized_infimnist, infimnist_labels, 0, bits_per_input);
    // print_binarized_image_raw(binarized_val, val_labels, 0, bits_per_input);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < num_train; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_train, sample_it), train_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %zu\n", sample_it);
    }

    uint64_t max_entry = 0;
    for(size_t discr_it = 0; discr_it < model.num_classes; ++discr_it) {
        for(size_t filter_it = 0; filter_it < model.num_filters; ++filter_it) {
            for(size_t entry_it = 0; entry_it < model.filter_entries; ++entry_it) {
                uint64_t el = *TENSOR3D(model.filters, discr_it, filter_it, entry_it);
                if(el > max_entry)
                    max_entry = el;
            }
        }
    }
    printf("Max entry: %llu\n", max_entry);

    printf("Testing to find optimal bleaching threshold\n");

    size_t best_bleach = 0;
    double best_accuracy = 0;
    for(size_t bleach = 1; bleach < 6; bleach+=1) {

        model.bleach = bleach;

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

    model.bleach = best_bleach;
    printf("Best bleach: %zu (%lf)\n", best_bleach, best_accuracy);

    printf("Accuracy on test set\n");
    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < num_test; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_test);
    printf("Test accuracy %zu/%zu (%f%%)\n", correct, num_test, 100 * accuracy);

    write_model("model.dat", &model);
}

int main(int argc, char *argv[]) {                              
    if(argc < 5) {
        printf("Error: usage: %s filter_inputs filter_entries filter_hashes bits_per_input block_size (0 for max size)\n", argv[0]);
        printf("\tExample usage: %s 28 1024 2 2 0\n", argv[0]);

        return 1;
    }

    size_t filter_inputs = atoi(argv[1]);
    size_t filter_entries = atoi(argv[2]);
    size_t filter_hashes = atoi(argv[3]);
    size_t bits_per_input = atoi(argv[4]);
    size_t block_size = atoi(argv[5]);

    printf("Training with parameters filter_inputs=%zu, filter_entries=%zu, filter_hashes=%zu, bits_per_input=%zu, block_size=%zu\n", filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size);

    train(filter_inputs, filter_entries, filter_hashes, bits_per_input, block_size);
}
