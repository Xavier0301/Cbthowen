SRC = $(wildcard src/*.c)
# SRC without main:
SRC_NOMAIN = $(filter-out src/main.c, $(wildcard src/*.c))
SRC_NOBINARIZER = $(filter-out src/binarizer.c, $(wildcard src/*.c))

.PHONY: all clean

all: main

main: $(SRC_NOBINARIZER)
	gcc -g $^ -o $@

binarizer: $(SRC_NOMAIN)
	gcc -g $^ -o $@

lib: $(SRC)
	gcc -g -fPIC -shared -o cbthowen.so $^

clean:
	rm -rf *.o *~ main
