module Data.Tensor where

import Data.Matrix

data Tensor t = Tensor {shape :: [Int], tenData :: [Matrix t]}
