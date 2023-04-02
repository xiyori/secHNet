{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Layers.Layer where

import Data.Matrix (Matrix)

class Layer t a where
    -- | layer -> input tensor -> (updated layer, output tensor)
    forward :: a -> Matrix t -> (a, Matrix t)
    -- | operation -> gradient over outputs ->
    --   updated layer -> gradient over inputs
    backward :: a -> Matrix t -> (a, Matrix t)

data Linear t = Linear {
  linearWeight :: Matrix t,
  linearGradWeight :: Matrix t,
  linearBias :: Matrix t,
  linearGradBias :: Matrix t,
  linearInput :: Matrix t
}

instance Layer t (Linear t) where
  forward :: Linear t -> Matrix t -> (Linear t, Matrix t)
  forward  = _

  backward :: Linear t -> Matrix t -> (Linear t, Matrix t)
  backward x = _

newtype ReLU t = ReLU {
  reluInput :: Matrix t
}

instance Layer t (ReLU t) where
  forward :: ReLU t -> Matrix t -> (ReLU t, Matrix t)
  forward  = _

  backward :: ReLU t -> Matrix t -> (ReLU t, Matrix t)
  backward x = _

data CrossEntropyLogits t = CrossEntropyLogits {
  сrossEntropyTarget :: Matrix t,
  сrossEntropyInput :: Matrix t
}

instance Layer t (CrossEntropyLogits t) where
  forward :: CrossEntropyLogits t -> Matrix t -> (CrossEntropyLogits t, Matrix t)
  forward  = _

  backward :: CrossEntropyLogits t -> Matrix t -> (CrossEntropyLogits t, Matrix t)
  backward x = _

setCrossEntropyTarget = _
