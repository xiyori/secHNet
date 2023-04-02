module Layers.Layer where

import Data.Matrix

class NNLayer t where
    -- input tensor -> output tensor
    forward :: Matrix t -> Matrix t
    -- input tensor -> output returned by Layer on forward call -> gradient over outputs -> gradient over inputs
    backward :: Matrix t -> Matrix t -> Matrix t -> Matrix t

