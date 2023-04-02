#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"


void train() {
    model_t model;

    size_t input_size = 784;
    size_t bits_per_input = 2;
    size_t num_inputs = input_size * bits_per_input;

    size_t num_classes = 10;

    size_t filter_inputs = 28;
    size_t filter_entries = 1024;
    size_t filter_hashes = 2;

    model_init(&model, num_inputs, num_classes, filter_inputs, filter_entries, filter_hashes);

    printf("Loading dataset\n");
    load_mnist();

    printf("Binarizing dataset\n");
    binarize_mnist(bits_per_input);

    print_binarized_mnist_image(7555, 2);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < MNIST_NUM_TRAIN; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_train, sample_it), train_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %zu\n", sample_it);
    }

    printf("Testing\n");
    model.bleach = 10; // optimal bleaching threshold

    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        uint64_t class = model_predict(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    printf("Accuracy %zu/%d (%f%%)\n", correct, MNIST_NUM_TEST, ((double) 100 * correct) / ((double) MNIST_NUM_TEST));
}

int main(int argc, char *argv[]) {                              

//   /* Error Checking */
//   if (argc < 2) {
//     printf("Error: usage: %s <program_file_1> <program_file_2> ...\n",
//            argv[0]);
//     exit(1);
//   }

    train();

    return 0;
}