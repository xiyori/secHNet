{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FunctionalDependencies #-}

module Data.Layers.Layer where

import Control.Applicative(liftA2)
import Data.Matrix (Matrix, elementwise, transpose,
                    fromLists, matrix, nrows, ncols, (!), zero, identity)

data Params t = Flat [t] | Nested [Params t]
instance Functor Params where
  fmap f (Flat x) = Flat $ map f x
  fmap f (Nested x) = Nested $ map (fmap f) x

instance Applicative Params where
    pure x = Flat [x]
    liftA2 f (Flat x) (Flat y) = Flat $ map (\(a, b) -> f a b) $ zip x y
    liftA2 f (Nested x) (Nested y) = Nested $ map (\(a, b) -> liftA2 f a b) $ zip x y
    liftA2 _ _ _ = error "Incorrect shape"

class Layer a t | a -> t where
  -- | layer -> input tensor -> (updated layer, output tensor)
  forward :: a -> Matrix t -> (a, Matrix t)
  -- | layer -> output gradient -> (updated layer, input gradient)
  backward :: a -> Matrix t -> (a, Matrix t)
  getParams :: a -> Params (Matrix t)
  setParams :: a -> Params (Matrix t) -> a
  getGrads :: a -> Params (Matrix t)
  setGrads :: a -> Params (Matrix t) -> a
    

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

makeLinear :: Int -> Int -> Linear Double
makeLinear x y = Linear (zero x y) (zero x y) (zero x 1) (zero x 1) (zero y 1)

instance Num t => Layer (Linear t) t where
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

  getParams x = Flat [linearWeight x, linearBias x]
  getGrads x = Flat [linearGradWeight x, linearGradBias x]
  setParams (Linear _ gradWeight _ gradBias input) (Flat l) = Linear (head l) gradWeight (l !! 1) gradBias input
  setGrads (Linear weight _ bias _ input) (Flat l) = Linear weight (head l) bias (l !! 1) input

newtype ReLU t = ReLU {
  reluInput :: Matrix t
}

makeReLU :: Int -> Int -> ReLU Double
makeReLU x y = ReLU (zero y 1)

instance (Ord t, Num t) => Layer (ReLU t) t where
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

  getParams _ = Flat []
  getGrads _ = Flat []
  setParams = const
  setGrads = const

data CrossEntropyLogits t = CrossEntropyLogits {
  -- | Class index from 1 to @n_classes@
  сrossEntropyTarget :: Int,
  -- | [n, 1]
  сrossEntropyInput :: Matrix t
}

makeCrossEntropyLogits :: Int -> Int -> CrossEntropyLogits Double
makeCrossEntropyLogits x y = CrossEntropyLogits 0 (zero y 1)

instance Floating t => Layer (CrossEntropyLogits t) t where
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
  
  getParams x = Flat []
  getGrads x = Flat []
  setParams = const
  setGrads = const
  

setCrossEntropyTarget :: CrossEntropyLogits t -> Int -> CrossEntropyLogits t
setCrossEntropyTarget (CrossEntropyLogits target input) newTarget =
  CrossEntropyLogits newTarget input
