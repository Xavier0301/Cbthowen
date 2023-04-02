# About

The repo hosts a C implementation of the paper:

**Weightless Neural Networks for Efficient Edge Inference**, Zachary Susskind, Aman Arora, Igor Dantas Dos Santos Miranda, Luis Armando Quintanilla Villon, Rafael Fontella Katopodis, Leandro Santiago de Araújo, Diego Leonel Cadette Dutra, Priscila Lima, Felipe Maia Galvão França, Mauricio Breternitz Jr., Lizy John. _Presented at the 31st International Conference on Parallel Architectures and Compilation Techniques (PACT 2022)_

_I am not affiliated with the research team that produced this paper_

1. See https://github.com/ZSusskind/BTHOWeN for the implementation by the authors

2. See https://github.com/Alantlb/wzero for a WNN library written in C++ with a python interface

# Running

Download the MNIST dataset (`t10k-images-idx3-ubyte`, ...) and place it in `data/`.

The `main` contains a simple example of loading the MNIST dataset, binarizing it, training the model and testing it.
``make
./main``

The library consists of the following parts:
1. A model that can be train and that can predict. Architecture is detailed in the above mentioned paper.
2. A data loader for the MNIST dataset
3. A way to apply Gaussian thermometer encoding to any dataset
4. A model manager (load/save)
5. A (very lightweight) tensor implementation

