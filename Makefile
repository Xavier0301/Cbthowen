SRC = $(wildcard src/*.c)
# SRC without main:
SRC_ONLYMAIN = $(filter-out src/benchmark.c, $(filter-out src/binarizer.c, $(filter-out src/trainer.c, $(SRC))))
SRC_ONLYBINARIZER = $(filter-out src/benchmark.c, $(filter-out src/main.c, $(filter-out src/trainer.c, $(SRC))))
SRC_ONLYTRAINER = $(filter-out src/benchmark.c, $(filter-out src/main.c, $(filter-out src/binarizer.c, $(SRC))))
SRC_ONLYBENCHMARK = $(filter-out src/trainer.c, $(filter-out src/main.c, $(filter-out src/binarizer.c, $(SRC))))

HASH ?= FAST_HASH
ENCODING ?= STRIDED_ENCODING
REORDERING ?= REORDER_SECOND

.PHONY: all clean

COMMON_FLAGS := -Wall -Wextra -g -lm -D${HASH} -D${ENCODING} -D${REORDERING}
CC := cc

all: main

main: $(SRC_ONLYMAIN)
	$(CC) ${COMMON_FLAGS} $^ -o $@

binarizer: $(SRC_ONLYBINARIZER)
	$(CC) ${COMMON_FLAGS} $^ -o $@

trainer: $(SRC_ONLYTRAINER)
	$(CC) ${COMMON_FLAGS} $^ -o $@

benchmark: $(SRC_ONLYBENCHMARK)
	$(CC) ${COMMON_FLAGS} $^ -o $@

lib: $(SRC_ONLYMAIN)
	$(CC) ${COMMON_FLAGS} -fPIC -shared -o cbthowen.so $^

clean:
	rm -rf *.o *~ main binarizer trainer benchmark cbthowen.so
