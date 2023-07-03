#!/bin/bash

declare -a BLOCK_SIZE_DIV=(1 2 4 8 16 32)

echo "Block sizes"
for i in ${!BLOCK_SIZE_DIV[@]}; do
    echo -n ${BLOCK_SIZE_DIV[$i]} ", "
done
echo ""

echo "MNIST-Small"
for i in ${!BLOCK_SIZE_DIV[@]}; do
    make clean &> /dev/null
    make trainer &> /dev/null
    echo -n ${BLOCK_SIZE_DIV[$i]} ", "
    ./trainer 28 1024 2 2 ${BLOCK_SIZE_DIV[$i]} 12 0 2> /dev/null | grep -a "test_accuracy"
done

echo "MNIST-Large"
for i in ${!BLOCK_SIZE_DIV[@]}; do
    make clean &> /dev/null
    make trainer &> /dev/null
    echo -n ${BLOCK_SIZE_DIV[$i]} ", "
    ./trainer 49 8192 4 6 ${BLOCK_SIZE_DIV[$i]} 5 0 2> /dev/null | grep -a "test_accuracy"
done

# NR_DPUS=1 NR_TASKLETS=16 PRINT=1 PERF=INSTRUCTION CHECK_RES=1 make

# NR_DPUS=2549 NR_TASKLETS=16 PRINT=0 PERF=INSTRUCTION CHECK_RES=0 make
