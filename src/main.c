#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "model_manager.h"

#include "batch.h"

void train() {
    printf("*TRAINING*\n");
    model_t model;

    size_t input_size = 784;
    size_t bits_per_input = 2;
    size_t num_inputs = input_size * bits_per_input;

    size_t num_classes = 10;

    size_t filter_inputs = 28;
    size_t filter_entries = 1024;
    size_t filter_hashes = 2;

    model_init(&model, num_inputs, num_classes, filter_inputs, filter_entries, filter_hashes, bits_per_input, 1);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    bmatrix_t test_patterns;
    bmatrix_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(&test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    bmatrix_t infimnist_patterns;
    bmatrix_init(&infimnist_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* infimnist_labels = calloc(num_train, sizeof(*infimnist_labels));
    load_infimnist(&infimnist_patterns, infimnist_labels, num_train);

    printf("Binarizing test dataset\n");
    bmatrix_t binarized_test;
    bmatrix_init(&binarized_test, num_test, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(&binarized_test, &test_patterns, MNIST_IM_SIZE, num_test, bits_per_input);

    // print_binarized_image(&binarized_test, test_labels, 0, 2);

    printf("Binarizing train dataset\n");
    bmatrix_t binarized_infimnist;
    bmatrix_init(&binarized_infimnist, num_train, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(&binarized_infimnist, &infimnist_patterns, MNIST_IM_SIZE, num_train, bits_per_input); 

    // print_binarized_image(&binarized_test, test_labels, 0, 2);
    print_binarized_image_raw(&binarized_infimnist, infimnist_labels, 0, 2);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < num_train; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_infimnist, sample_it), infimnist_labels[sample_it]);
        if(sample_it % 10000 == 0)
            printf("    %zu\n", sample_it);
    }

    uint64_t max_entry = 0;
    for(size_t discr_it = 0; discr_it < model.num_classes; ++discr_it) {
        for(size_t filter_it = 0; filter_it < model.num_filters; ++filter_it) {
            for(size_t entry_it = 0; entry_it < model.filter_entries; ++entry_it) {
                uint64_t el = *TENSOR3D(model.data, discr_it, filter_it, entry_it);
                if(el > max_entry)
                    max_entry = el;
            }
        }
    }
    printf("Max entry: %llu\n", max_entry);

    printf("Testing to find optimal bleaching threshold\n");

    size_t best_bleach = 0;
    double best_accuracy = 0;
    for(size_t bleach = 1; bleach < 20; ++bleach) {

        model.bleach = bleach;

        size_t correct = 0;
        for(size_t sample_it = 0; sample_it < num_test; ++sample_it) {
            size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
            correct += (class == test_labels[sample_it]);
        }

        double accuracy = ((double) correct) / ((double) num_test);
        printf("Bleach %zu. Accuracy %zu/%zu (%f%%)\n", bleach, correct, num_test, 100 * accuracy);

        if(accuracy >= best_accuracy) {
            best_bleach = bleach;
            best_accuracy = accuracy;
        }
    }

    model.bleach = best_bleach;
    printf("Best bleach: %zu (%lf)\n", best_bleach, best_accuracy);

    write_model("model.dat", &model);
}

void load_and_test() {
    printf("*LOADING AND TESTING MODEL*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    bmatrix_t test_patterns;
    bmatrix_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(&test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    bmatrix_t train_patterns;
    bmatrix_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_mnist_train(&train_patterns, train_labels, num_train);

    printf("Binarizing test dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_test;
    bmatrix_init(&binarized_test, num_test, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_test, &test_patterns, MNIST_IM_SIZE, num_test, model.bits_per_input);

    printf("Binarizing train dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_train;
    bmatrix_init(&binarized_train, num_train, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_train, &train_patterns, MNIST_IM_SIZE, num_train, model.bits_per_input);

    print_binarized_image_raw(&binarized_train, train_labels, 0, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < num_train; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_train, sample_it));
        correct += (class == train_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_train);
    printf("Accuracy %zu/%zu (%f%%)\n", correct, num_train, 100 * accuracy);

}

void compare() {
    printf("*COMPARING PREDICT1 AND PREDICT2*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    bmatrix_t test_patterns;
    bmatrix_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(&test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    bmatrix_t train_patterns;
    bmatrix_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_mnist_train(&train_patterns, train_labels, num_train);

    printf("Binarizing test dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_test;
    bmatrix_init(&binarized_test, num_test, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_test, &test_patterns, MNIST_IM_SIZE, num_test, model.bits_per_input);

    printf("Binarizing train dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_train;
    bmatrix_init(&binarized_train, num_train, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_train, &train_patterns, MNIST_IM_SIZE, num_train, model.bits_per_input);

    print_binarized_image_raw(&binarized_train, train_labels, 0, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t agree = 0;
    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        size_t class1 = model_predict(&model, MATRIX_AXIS1(binarized_test, sample_it));
        size_t class2 = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        agree += (class1 == class2);
    }
    printf("Agreeing: %lf%%\n", 100 * ((double) agree) / MNIST_NUM_TEST);
}

void test_batching() {
    printf("*TEST BATCHING*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    bmatrix_t test_patterns;
    bmatrix_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(&test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    bmatrix_t train_patterns;
    bmatrix_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_mnist_train(&train_patterns, train_labels, num_train);

    printf("Binarizing test dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_test;
    bmatrix_init(&binarized_test, num_test, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_test, &test_patterns, MNIST_IM_SIZE, num_test, model.bits_per_input);

    printf("Binarizing train dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_train;
    bmatrix_init(&binarized_train, num_train, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_train, &train_patterns, MNIST_IM_SIZE, num_train, model.bits_per_input);

    print_binarized_image_raw(&binarized_train, train_labels, 0, 2);

    printf("Testing with bleach %d\n", model.bleach);

    size_t batch_size = 30;
    size_t agree = 0;

    size_t* results = calloc(batch_size, sizeof(*results));
    size_t* results_batch = calloc(batch_size, sizeof(*results_batch));

    batch_prediction(results_batch, &model, &binarized_test, batch_size);

    for(size_t sample_it = 0; sample_it < batch_size; ++sample_it) {
        results[sample_it] = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        printf("Sample %zu: %zu\n", sample_it, results[sample_it]);
        agree += (results[sample_it] == results_batch[sample_it]);
    }
    printf("Agreeing: %lf%%\n", 100 * ((double) agree) / batch_size);
}

void test_reordering_dataset() {
    printf("*TEST REORDERING DATASET*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    bmatrix_t test_patterns;
    bmatrix_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(&test_patterns, test_labels, num_test);

    printf("Binarizing test dataset with %zu bits per input\n", model.bits_per_input);
    bmatrix_t binarized_test;
    bmatrix_init(&binarized_test, num_test, MNIST_IM_SIZE * model.bits_per_input);
    binarize_matrix(&binarized_test, &test_patterns, MNIST_IM_SIZE, num_test, model.bits_per_input);

    print_binarized_image_raw(&binarized_test, test_labels, 0, 2);

    printf("Reordering dataset\n");
    bmatrix_t reordered_binarized_test;
    bmatrix_init(&reordered_binarized_test, num_test, MNIST_IM_SIZE * model.bits_per_input);
    reorder_dataset(&reordered_binarized_test, &binarized_test, model.input_order, num_test, model.num_inputs_total);

    printf("Testing with bleach %d\n", model.bleach);

    size_t agree = 0;

    unsigned char* reordered_sample = (unsigned char *) calloc(MNIST_IM_SIZE * model.bits_per_input, sizeof(*binarized_test.data));

    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        reorder_array(reordered_sample, MATRIX_AXIS1(binarized_test, sample_it), model.input_order, model.num_inputs_total);
        
        if(memcmp(reordered_sample, MATRIX_AXIS1(reordered_binarized_test, sample_it), MNIST_IM_SIZE * model.bits_per_input * sizeof(*reordered_binarized_test.data)) == 0) {
            agree += 1;
        }
    }
    printf("Reorder agreeing: %lf%%\n", 100 * ((double) agree) / MNIST_NUM_TEST);
}

void print_model_data() {
    printf("*PRINTING MODEL DATA*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    for(size_t discr_it = 0; discr_it < 1; ++discr_it) {
        printf("Discriminator %zu.\n\n", discr_it);
        for(size_t filter_it = 0; filter_it < 1; ++filter_it) {
            printf("Filter %zu.\n", filter_it);
            for(size_t entry_it = 0; entry_it < model.filter_entries; ++entry_it) {
                uint32_t el = *TENSOR3D(model.data, discr_it, filter_it, entry_it);
                printf("%u ", el);
            }
            printf("\n");
        }
    }
}

int main(int argc, char *argv[]) {                              

    /* Error Checking */
    if(argc < 2) {
        printf("Error: usage: %s 0..5.\n\t \
        0 is for training from scratch\n\t \
        1 is for loading model.dat and testing\n\t \
        2 is for comparing predict1 and predict2\n\t \
        3 is for testing batching\n\t \
        4 if for testing reordering dataset\n\t \
        5 is for printing model data\n",
                argv[0]);
        exit(1);
    }

    if(argv[1][0] == '0')
        train();
    else if(argv[1][0] == '1')
        load_and_test();
    else if(argv[1][0] == '2')
        compare();
    else if(argv[1][0] == '3')
        test_batching();
    else if(argv[1][0] == '4')
        test_reordering_dataset();
    else 
        print_model_data();

    return 0;
}
