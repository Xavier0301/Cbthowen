SRC = $(wildcard src/*.c)

.PHONY: all verify clean

all: main

main: $(SRC)
	gcc -g $^ -o $@

lib: $(SRC)
	gcc -g -fPIC -shared -o cbthowen.so $^

clean:
	rm -rf *.o *~ main
