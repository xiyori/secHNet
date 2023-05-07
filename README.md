# Autograd and Neural Network Library in Haskell

**Key features**

+ *Efficient implementation of tensors with vectorized operations, O(1) slicing and more*
+ *Fliexible computational graph supports NN architectures like ResNet*
+ *Efficient convolutions via stride trick (TODO)*
+ *A wide variety of optimizers (TODO)*
+ *Highly customizable dataloader*
+ *Simple training process (TODO)*
+ *Comprehensible usage manual (TODO)*
+ *QuickCheck CI*

**Install**

`sudo apt install libopenblas-dev`

**Compile**

Compile with optimizations (slow compile time, fast execution):

`cabal v2-run -O2 sechnet`

Compile with profiling:

`cabal v2-run -O2 --enable-profiling sechnet -- +RTS -p`

Quick compile without optimizations:

`cabal v2-run -O0 sechnet`

QuickCheck CI:

`cabal v2-test -O0 sechnet-test`
