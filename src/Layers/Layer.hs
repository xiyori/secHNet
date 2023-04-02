{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Layers.Layer where

import Data.Matrix (Matrix, elementwise, transpose,
                    fromLists, matrix, nrows, ncols, (!))

class Layer t a where
  -- | layer -> input tensor -> (updated layer, output tensor)
  forward :: a -> Matrix t -> (a, Matrix t)
  -- | operation -> gradient over outputs ->
  --   (updated layer, gradient over inputs)
  backward :: a -> Matrix t -> (a, Matrix t)

class HasParams t a where
    getParams :: a -> [Matrix t]
    setParams :: a -> [Matrix t] -> a
    getGrads :: a -> [Matrix t]
    setGrads :: a -> [Matrix t] -> a

data Linear t = Linear {
  -- | [n, m]
  linearWeight :: Matrix t,
  -- | [n, m]
  linearGradWeight :: Matrix t,
  -- | [n, 1]
  linearBias :: Matrix t,
  -- | [n, 1]
  linearGradBias :: Matrix t,
  -- | [m, 1]
  linearInput :: Matrix t
}

instance Num t => Layer t (Linear t) where
  forward :: Linear t -> Matrix t -> (Linear t, Matrix t)
  forward (Linear weight gradWeight bias gradBias _) input =
    (Linear weight gradWeight bias gradBias input, weight * input + bias)

  backward :: Linear t -> Matrix t -> (Linear t, Matrix t)
  backward (Linear weight gradWeight bias gradBias input) grad_output =
    (Linear weight (gradWeight + newGradWeight)
    bias (gradBias + newGradBias) input, grad_input)
      where
        grad_input = transpose weight * grad_output
        newGradWeight = grad_output * transpose input
        newGradBias = grad_output

newtype ReLU t = ReLU {
  reluInput :: Matrix t
}

instance (Ord t, Num t) => Layer t (ReLU t) where
  forward :: ReLU t -> Matrix t -> (ReLU t, Matrix t)
  forward _ input = (ReLU input, fmap relu input)
    where
      relu x
        | x > 0     = x
        | otherwise = 0

  backward :: ReLU t -> Matrix t -> (ReLU t, Matrix t)
  backward relu@(ReLU input) grad_output =
    (relu, elementwise zeroMask grad_output input)
      where
        zeroMask x mask
          | mask > 0  = x
          | otherwise = 0

data CrossEntropyLogits t = CrossEntropyLogits {
  -- | Class index from 1 to @n_classes@
  сrossEntropyTarget :: Int,
  -- | [n, 1]
  сrossEntropyInput :: Matrix t
}

instance Floating t => Layer t (CrossEntropyLogits t) where
  forward :: CrossEntropyLogits t -> Matrix t -> (CrossEntropyLogits t, Matrix t)
  forward (CrossEntropyLogits target _) logits =
    (CrossEntropyLogits target logits,
    fromLists [[-(logits ! (target, 1)) + (log . sum . fmap exp) logits]])

  backward :: CrossEntropyLogits t -> Matrix t -> (CrossEntropyLogits t, Matrix t)
  backward crossEntropy@(CrossEntropyLogits target logits) _ =
    (crossEntropy, -targetMask + softmax)
      where
        targetMask = matrix (nrows logits) (ncols logits) setClasses
        setClasses (i, _)
          | i == target = 1
          | otherwise   = 0
        softmax = fmap (divideBySum . exp) logits
        divideBySum = (/ (sum . fmap exp) logits)

setCrossEntropyTarget :: CrossEntropyLogits t -> Int -> CrossEntropyLogits t
setCrossEntropyTarget (CrossEntropyLogits target input) newTarget =
  CrossEntropyLogits newTarget input
