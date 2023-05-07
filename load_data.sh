#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz --output-document data.tar.gz
tar -xf data.tar.gz
mv cifar-10-batches-bin data
rm data.tar.gz
