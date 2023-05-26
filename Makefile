SRC = $(wildcard src/*.c)
# SRC without main:
SRC_NOMAIN = $(filter-out src/main.c, $(wildcard src/*.c))
SRC_NOBINARIZER = $(filter-out src/binarizer.c, $(wildcard src/*.c))

.PHONY: all clean

COMMON_FLAGS := -Wall -Wextra -g
CC := cc

all: main

main: $(SRC_NOBINARIZER)
	$(CC) ${COMMON_FLAGS} $^ -o $@

binarizer: $(SRC_NOMAIN)
	$(CC) ${COMMON_FLAGS} $^ -o $@

lib: $(SRC)
	$(CC) ${COMMON_FLAGS} -fPIC -shared -o cbthowen.so $^

clean:
	rm -rf *.o *~ main
