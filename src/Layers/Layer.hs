{-# LANGUAGE MultiParamTypeClasses #-}

module Layers.Layer where

import Data.Matrix

class Layer t a where
    -- input tensor -> output tensor
    forward :: a -> Matrix t -> Matrix t
    -- input tensor -> output returned by Layer on forward call -> gradient over outputs -> gradient over inputs
    backward :: a -> Matrix t -> Matrix t -> Matrix t -> Matrix t

