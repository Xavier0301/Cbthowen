SRC = $(wildcard src/*.c)
# SRC without main:
SRC_ONLYMAIN = $(filter-out src/binarizer.c, $(filter-out src/trainer.c, $(SRC)))
SRC_ONLYBINARIZER = $(filter-out src/main.c, $(filter-out src/trainer.c, $(SRC)))
SRC_ONLYTRAINER = $(filter-out src/main.c, $(filter-out src/binarizer.c, $(SRC)))

.PHONY: all clean

COMMON_FLAGS := -Wall -Wextra -g -lm
CC := cc

all: main

main: $(SRC_ONLYMAIN)
	$(CC) ${COMMON_FLAGS} $^ -o $@

binarizer: $(SRC_ONLYBINARIZER)
	$(CC) ${COMMON_FLAGS} $^ -o $@

trainer: $(SRC_ONLYTRAINER)
	$(CC) ${COMMON_FLAGS} $^ -o $@

lib: $(SRC)
	$(CC) ${COMMON_FLAGS} -fPIC -shared -o cbthowen.so $^

clean:
	rm -rf *.o *~ main binarizer trainer cbthowen.so
