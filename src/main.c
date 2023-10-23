#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "types.h"

#include "model.h"
#include "tensor.h"
#include "data_loader.h"
#include "data_manager.h"

#include "batch.h"

void train() {
    printf("*TRAINING*\n");
    model_t model;

    model_init_params_t params = {
        .num_classes = 10,

        .num_inputs = MNIST_IM_SIZE * 2,
        .bits_per_input = 2,

        .dim1_block_size = 1,
        .dim2_block_size = 1,

        .filter_hashes = 2,
        .filter_inputs = 28,
        .filter_entries = 1024,
    };

    model_init(&model, &params);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    mat_u8 infimnist_patterns;
    matrix_u8_init(&infimnist_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* infimnist_labels = calloc(num_train, sizeof(*infimnist_labels));
    load_infimnist(infimnist_patterns, infimnist_labels, num_train);

    printf("Binarizing test dataset\n");
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * params.bits_per_input);
    binarize_matrix(binarized_test, test_patterns, MNIST_IM_SIZE, num_test, params.bits_per_input);

    // print_binarized_image(&binarized_test, test_labels, 0, 2);

    printf("Binarizing train dataset\n");
    mat_u8 binarized_infimnist;
    matrix_u8_init(&binarized_infimnist, num_train, MNIST_IM_SIZE * params.bits_per_input);
    binarize_matrix(binarized_infimnist, infimnist_patterns, MNIST_IM_SIZE, num_train, params.bits_per_input); 

    // print_binarized_image(&binarized_test, test_labels, 0, 2);
    print_binarized_image_raw(binarized_infimnist, infimnist_labels, 0, 2);

    printf("Training\n");

    for(size_t sample_it = 0; sample_it < num_train; ++sample_it) {
        model_train(&model, MATRIX_AXIS1(binarized_infimnist, sample_it), infimnist_labels[sample_it]);
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
    for(size_t bleach = 1; bleach < 20; ++bleach) {

        model.p.bleach = bleach;

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

    model.p.bleach = best_bleach;
    printf("Best bleach: %zu (%lf)\n", best_bleach, best_accuracy);

    write_model("model.dat", &model);
}

void load_and_test() {
    printf("*LOADING AND TESTING MODEL*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    size_t num_test = MNIST_NUM_TEST;
    printf("Loading test dataset (%zu)\n", num_test);
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    size_t num_total = MNIST_NUM_TRAIN;

    printf("Loading total train dataset (%zu)\n", num_total);
    mat_u8 tmp_train_patterns;
    matrix_u8_init(&tmp_train_patterns, num_total, MNIST_IM_SIZE);
    unsigned char* tmp_train_labels = calloc(num_total, sizeof(*tmp_train_labels));
    load_infimnist(tmp_train_patterns, tmp_train_labels, num_total);

    printf("Binarizing test dataset\n"); 
    printf("\t1. Calculate mean/var of train dset\n");
    double mean[MNIST_IM_SIZE];
    double variance[MNIST_IM_SIZE];

    mat_u8_mean(mean, tmp_train_patterns, MNIST_IM_SIZE, num_total);
    mat_u8_variance(variance, tmp_train_patterns, MNIST_IM_SIZE, num_total, mean);

    printf("\t2. Binarizing\n");
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix_meanvar(
        binarized_test, 
        test_patterns, 
        mean, variance,
        num_test, model.p.bits_per_input);


    printf("Testing with bleach %u\n", model.p.bleach);

    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < num_test; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        correct += (class == test_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_test);
    printf("Accuracy %zu/%zu (%.2f%%)\n", correct, num_test, 100 * accuracy);

}

void compare() {
    printf("*COMPARING PREDICT1 AND PREDICT2*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading test dataset\n");
    size_t num_test = MNIST_NUM_TEST;
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    mat_u8 train_patterns;
    matrix_u8_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_mnist_train(train_patterns, train_labels, num_train);

    printf("Binarizing test dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_test, test_patterns, MNIST_IM_SIZE, num_test, model.p.bits_per_input);

    printf("Binarizing train dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_train;
    matrix_u8_init(&binarized_train, num_train, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_train, train_patterns, MNIST_IM_SIZE, num_train, model.p.bits_per_input);

    print_binarized_image_raw(binarized_train, train_labels, 0, 2);

    printf("Testing with bleach %u\n", model.p.bleach);

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
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    printf("Loading train dataset\n");
    size_t num_train = MNIST_NUM_TRAIN;
    mat_u8 train_patterns;
    matrix_u8_init(&train_patterns, num_train, MNIST_IM_SIZE);
    unsigned char* train_labels = calloc(num_train, sizeof(*train_labels));
    load_mnist_train(train_patterns, train_labels, num_train);

    printf("Binarizing test dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_test, test_patterns, MNIST_IM_SIZE, num_test, model.p.bits_per_input);

    printf("Binarizing train dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_train;
    matrix_u8_init(&binarized_train, num_train, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_train, train_patterns, MNIST_IM_SIZE, num_train, model.p.bits_per_input);

    print_binarized_image_raw(binarized_train, train_labels, 0, 2);

    printf("Testing with bleach %u\n", model.p.bleach);

    size_t batch_size = 30;
    size_t agree = 0;

    size_t* results = calloc(batch_size, sizeof(*results));
    size_t* results_batch = calloc(batch_size, sizeof(*results_batch));

    batch_prediction(results_batch, &model, binarized_test, batch_size);

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
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    printf("Binarizing test dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_test, test_patterns, MNIST_IM_SIZE, num_test, model.p.bits_per_input);

    print_binarized_image_raw(binarized_test, test_labels, 0, 2);

    printf("Reordering dataset\n");
    mat_u8 reordered_binarized_test;
    matrix_u8_init(&reordered_binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    reorder_dataset(reordered_binarized_test, binarized_test, model.input_order, num_test, model.p.num_inputs_total);

    printf("Testing with bleach %u\n", model.p.bleach);

    size_t agree = 0;

    unsigned char* reordered_sample = (unsigned char *) calloc(MNIST_IM_SIZE * model.p.bits_per_input, sizeof(*binarized_test.data));

    for(size_t sample_it = 0; sample_it < MNIST_NUM_TEST; ++sample_it) {
        reorder_array(reordered_sample, MATRIX_AXIS1(binarized_test, sample_it), model.input_order, model.p.num_inputs_total);
        
        if(memcmp(reordered_sample, MATRIX_AXIS1(reordered_binarized_test, sample_it), MNIST_IM_SIZE * model.p.bits_per_input * sizeof(*reordered_binarized_test.data)) == 0) {
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
            for(size_t entry_it = 0; entry_it < model.p.filter_entries; ++entry_it) {
                u32 el = *TENSOR3D(model.filters, discr_it, filter_it, entry_it);
                printf("%u ", el);
            }
            printf("\n");
        }
    }
}

void binarize_and_save() {
    printf("*BINARIZING AND SAVING*\n");
    model_t model;

    printf("Loading model\n");

    read_model("model.dat", &model);

    printf("Loading infimnist\n"); 
    size_t num_samples = MNIST_NUM_TRAIN;
    mat_u8 infimnist_patterns;
    matrix_u8_init(&infimnist_patterns, num_samples, MNIST_IM_SIZE);
    unsigned char* infimnist_labels = calloc(num_samples, sizeof(*infimnist_labels));
    load_infimnist(infimnist_patterns, infimnist_labels, num_samples);

    printf("Binarizing infimnist dataset with %u bits per input\n", model.p.bits_per_input);
    mat_u8 binarized_infimnist;
    matrix_u8_init(&binarized_infimnist, num_samples, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix(binarized_infimnist, infimnist_patterns, MNIST_IM_SIZE, num_samples, model.p.bits_per_input);

    printf("Saving binarized infimnist\n");
    write_dataset("binarized_infimnist.dat", binarized_infimnist, num_samples, MNIST_IM_SIZE * model.p.bits_per_input);

    printf("Loading binarized infimnist into another buffer\n");
    mat_u8 binarized_infimnist2;
    matrix_u8_init(&binarized_infimnist2, num_samples, MNIST_IM_SIZE * model.p.bits_per_input);
    size_t num_samples_, sample_size_;
    read_dataset("binarized_infimnist.dat", binarized_infimnist2, &num_samples_, &sample_size_);
    assert(num_samples_ == num_samples);
    assert(sample_size_ == MNIST_IM_SIZE * model.p.bits_per_input);

    printf("Veryifying that the two buffers are the same\n");
    size_t agree = 0;
    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
        if(memcmp(MATRIX_AXIS1(binarized_infimnist, sample_it), MATRIX_AXIS1(binarized_infimnist2, sample_it), MNIST_IM_SIZE * model.p.bits_per_input * sizeof(*binarized_infimnist.data)) == 0) {
            agree += 1;
        }
    }
    printf("Agreeing: %lf%%\n", 100 * ((double) agree) / num_samples);
}

void load_and_print_binarized() {
    printf("*LOADING AND PRINTING BINARIZED*\n");

    // Load model
    size_t bits_per_input = 2;
    printf("Bits per input %zu\n", bits_per_input);
         
    printf("Loading infimnist\n"); 
    size_t num_samples = MNIST_NUM_TRAIN;
    mat_u8 infimnist_patterns;
    matrix_u8_init(&infimnist_patterns, num_samples, MNIST_IM_SIZE);
    unsigned char* infimnist_labels = calloc(num_samples, sizeof(*infimnist_labels));
    load_infimnist(infimnist_patterns, infimnist_labels, num_samples);

    printf("Binarizing infimnist dataset with %zu bits per input\n", bits_per_input);
    mat_u8 binarized_infimnist;
    matrix_u8_init(&binarized_infimnist, num_samples, MNIST_IM_SIZE * bits_per_input);
    binarize_matrix(binarized_infimnist, infimnist_patterns, MNIST_IM_SIZE, num_samples, bits_per_input);

    for(size_t idx = 0; idx < 5; ++idx)
        print_binarized_image_raw(binarized_infimnist, infimnist_labels, idx, bits_per_input);
}

void infer_from_binarized_dset() {
    printf("*INFER FROM BINARIZED DSET*\n");

    // Load model
    printf("Loading model\n");
         
    model_t model;
    read_model("model.dat", &model);

    print_model_params(&model.p);
    
    // Loading binarized dataset
    printf("Loading dataset\n");
    const unsigned int num_samples = 100000;
    mat_u8 binarized_infimnist;
    matrix_u8_init(&binarized_infimnist, num_samples, MNIST_IM_SIZE * model.p.bits_per_input);
    size_t num_samples_total, sample_size;
    read_dataset_partial("./data/binarized8m.dat", binarized_infimnist, num_samples, &num_samples_total, &sample_size);

    // Loading labels
    printf("Loading labels\n");
    unsigned char* infimnist_labels = calloc(num_samples, sizeof(*infimnist_labels));
    load_infimnist_labels(infimnist_labels, num_samples);

    print_binarized_image_raw(binarized_infimnist, infimnist_labels, 0, 2);

    // Inference
    printf("Inference\n");
    size_t correct = 0;
    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it) {
        size_t class = model_predict2(&model, MATRIX_AXIS1(binarized_infimnist, sample_it));
        correct += (class == infimnist_labels[sample_it]);
    }

    double accuracy = ((double) correct) / ((double) num_samples);
    printf("Accuracy %zu/%u (%f%%)\n", correct, num_samples, 100 * accuracy);
}

void load_mnist() {
    printf("*LOADING MNIST*\n");

    size_t num_samples = MNIST_NUM_TRAIN;
    mat_u8 infimnist_patterns;
    matrix_u8_init(&infimnist_patterns, num_samples, MNIST_IM_SIZE);
    unsigned char* infimnist_labels = calloc(num_samples, sizeof(*infimnist_labels));
    load_infimnist(infimnist_patterns, infimnist_labels, num_samples);
}

void test_sparsity() {
    printf("*TEST SPARSITY*\n");

    // Load model
    printf("Loading model\n");
         
    model_t model;
    read_model("model.dat", &model);

    model_bleach(&model);

    size_t total_entries = model.p.num_classes * model.p.num_filters * model.p.filter_entries;
    size_t num_nonzero = 0;
    for(size_t discr_it = 0; discr_it < model.p.num_classes; ++discr_it) {
        for(size_t filter_it = 0; filter_it < model.p.num_filters; ++filter_it) {
            for(size_t entry_it = 0; entry_it < model.p.filter_entries; ++entry_it) {
                u16 entry = *TENSOR3D(model.filters, discr_it, filter_it, entry_it);
                if(entry != 0)
                    num_nonzero += 1;
            }
        }
    }

    printf("Sparsity: %lf%%\n", 100 * ((double) num_nonzero) / total_entries);
}

void test_packed() {
    printf("*TEST PACKED*\n");

    // Load model
    printf("Loading model\n");
         
    model_t model;
    read_model("model.dat", &model);

    model_bleach(&model);
    pmodel_t pmodel;
    model_pack(&model, &pmodel);

    size_t num_test = MNIST_NUM_TEST;
    printf("Loading test dataset (%zu)\n", num_test);
    mat_u8 test_patterns;
    matrix_u8_init(&test_patterns, num_test, MNIST_IM_SIZE);
    unsigned char* test_labels = calloc(num_test, sizeof(*test_labels));
    load_mnist_test(test_patterns, test_labels, num_test);

    size_t num_total = MNIST_NUM_TRAIN;

    printf("Loading total train dataset (%zu)\n", num_total);
    mat_u8 tmp_train_patterns;
    matrix_u8_init(&tmp_train_patterns, num_total, MNIST_IM_SIZE);
    unsigned char* tmp_train_labels = calloc(num_total, sizeof(*tmp_train_labels));
    load_infimnist(tmp_train_patterns, tmp_train_labels, num_total);

    printf("Binarizing test dataset\n"); 
    printf("\t1. Calculate mean/var of train dset\n");
    double mean[MNIST_IM_SIZE];
    double variance[MNIST_IM_SIZE];

    mat_u8_mean(mean, tmp_train_patterns, MNIST_IM_SIZE, num_total);
    mat_u8_variance(variance, tmp_train_patterns, MNIST_IM_SIZE, num_total, mean);

    printf("\t2. Binarizing\n");
    mat_u8 binarized_test;
    matrix_u8_init(&binarized_test, num_test, MNIST_IM_SIZE * model.p.bits_per_input);
    binarize_matrix_meanvar(
        binarized_test, 
        test_patterns, 
        mean, variance,
        num_test, model.p.bits_per_input);

    printf("Testing\n");
    size_t correct = 0;
    size_t pcorrect = 0;
    size_t matching = 0;
    for(size_t sample_it = 0; sample_it < num_test; ++sample_it) {
        u64 class = model_predict2(&model, MATRIX_AXIS1(binarized_test, sample_it));
        u64 pclass = pmodel_predict(&pmodel, MATRIX_AXIS1(binarized_test, sample_it));
        // printf("\n");
        correct += (class == test_labels[sample_it]);
        pcorrect += (pclass == test_labels[sample_it]);
        matching += (class == pclass);
    }

    double accuracy = ((double) correct) / ((double) num_test);
    double paccuracy = ((double) pcorrect) / ((double) num_test);
    double matching_accuracy = ((double) matching) / ((double) num_test);

    printf("Accuracies:\n");
    printf("\tbase %zu/%zu (%.2lf%%)\n", correct, num_test, 100 * accuracy);
    printf("\tpacked %zu/%zu(%.2lf%%)\n", pcorrect, num_test, 100 * paccuracy);
    printf("Matching:\n\t%zu/%zu (%.2lf%%)\n", matching, num_test, 100 * matching_accuracy);
}

int main(int argc, char *argv[]) {  

// Put the error message in a char array:
    const char error_message[] = "Error: usage: %s X\n\t \
        0 is for training from scratch\n\t \
        1 is for loading model.dat and testing\n\t \
        2 is for comparing predict1 and predict2\n\t \
        3 is for testing batching\n\t \
        4 if for testing reordering dataset\n\t \
        5 is for printing model data\n\t \
        6 is for binarizing and saving infimnist\n\t \
        7 is for loading and printing binarized infimnist\n\t \
        8 is for infering from binarized infimnist\n\t \
        9 is for loading mnist\n\t \
        a is for testing sparsity\n\t \
        b is for testing packed\n";


    /* Error Checking */
    if(argc < 2) {
        printf(error_message, argv[0]);
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
    else if(argv[1][0] == '5')
        print_model_data();
    else if(argv[1][0] == '6')
        binarize_and_save();
    else if(argv[1][0] == '7')
        load_and_print_binarized();
    else if(argv[1][0] == '8')
        infer_from_binarized_dset();
    else if(argv[1][0] == '9')
        load_mnist();
    else if(argv[1][0] == 'a')
        test_sparsity();
    else if(argv[1][0] == 'b')
        test_packed();
    else {
        printf(error_message, argv[0]);
        exit(1);
    }

    return 0;
}
