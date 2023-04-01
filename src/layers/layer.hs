{-# LANGUAGE MultiParamTypeClasses #-}

module NN.Layer where

import Data.Tensor

class NNLayer n i o where
    -- input tensor -> output tensor
    forward :: Tensor i n -> Tensor o n
    -- input tensor -> output returned by Layer on forward call -> gradient over outputs -> gradient over inputs
    backward :: Tensor i n -> Tensor o n -> Tensor o n -> Tensor i n

