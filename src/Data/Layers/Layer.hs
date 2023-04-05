{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FunctionalDependencies #-}

module Data.Layers.Layer where

import Control.Applicative(liftA2)
import Data.Tensor

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
  forward :: a -> Tensor t -> (a, Tensor t)
  -- | layer -> output gradient -> (updated layer, input gradient)
  backward :: a -> Tensor t -> (a, Tensor t)
  getParams :: a -> Params (Tensor t)
  setParams :: a -> Params (Tensor t) -> a
  getGrads :: a -> Params (Tensor t)
  setGrads :: a -> Params (Tensor t) -> a


data Conv2d t = Conv2d {
  -- | [n, m]
  conv2dWeight :: Tensor t,
  -- | [n, m]
  conv2dGradWeight :: Tensor t,
  -- | [n, 1]
  conv2dBias :: Tensor t,
  -- | [n, 1]
  conv2dGradBias :: Tensor t,
  -- | [m, 1]
  conv2dInput :: Tensor t
}

-- | In channels -> out channels -> kernel_size -> stride -> padding
makeConv2d :: Int -> Int -> Int -> Int -> Int -> Conv2d Double
makeConv2d inChannels outChannels kernel stride padding = _

instance Num t => Layer (Conv2d t) t where
  forward :: Conv2d t -> Tensor t -> (Conv2d t, Tensor t)
  forward = _

  backward :: Conv2d t -> Tensor t -> (Conv2d t, Tensor t)
  backward = _

  getParams x = Flat [conv2dWeight x, conv2dBias x]
  getGrads x = Flat [conv2dGradWeight x, conv2dGradBias x]
  setParams (Conv2d _ gradWeight _ gradBias input) (Flat l) =
    Conv2d (head l) gradWeight (l !! 1) gradBias input
  setGrads (Conv2d weight _ bias _ input) (Flat l) =
    Conv2d weight (head l) bias (l !! 1) input


data Linear t = Linear {
  -- | [n, m]
  linearWeight :: Tensor t,
  -- | [n, m]
  linearGradWeight :: Tensor t,
  -- | [n, 1]
  linearBias :: Tensor t,
  -- | [n, 1]
  linearGradBias :: Tensor t,
  -- | [m, 1]
  linearInput :: Tensor t
}

makeLinear :: Int -> Int -> Linear Double
makeLinear i o = _

instance Num t => Layer (Linear t) t where
  forward :: Linear t -> Tensor t -> (Linear t, Tensor t)
  forward (Linear weight gradWeight bias gradBias _) input =
    (Linear weight gradWeight bias gradBias input, weight * input + bias)

  backward :: Linear t -> Tensor t -> (Linear t, Tensor t)
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
  reluInput :: Tensor t
}

makeReLU :: Int -> ReLU Double
makeReLU i = ReLU (zeros [i, 1])

instance (Ord t, Num t) => Layer (ReLU t) t where
  forward :: ReLU t -> Tensor t -> (ReLU t, Tensor t)
  forward _ input = (ReLU input, fmap relu input)
    where
      relu x
        | x > 0     = x
        | otherwise = 0

  backward :: ReLU t -> Tensor t -> (ReLU t, Tensor t)
  backward relu@(ReLU input) grad_output =
    (relu, performWithBroadcasting zeroMask grad_output input)
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
  сrossEntropyInput :: Tensor t
}

makeCrossEntropyLogits :: Int -> Int -> CrossEntropyLogits Double
makeCrossEntropyLogits label i = CrossEntropyLogits label (zero i 1)

instance Floating t => Layer (CrossEntropyLogits t) t where
  forward :: CrossEntropyLogits t -> Tensor t -> (CrossEntropyLogits t, Tensor t)
  forward (CrossEntropyLogits target _) logits =
    (CrossEntropyLogits target logits,
    fromLists [[-(logits ! (target + 1, 1)) + (log . sum . fmap exp) logits]])

  backward :: CrossEntropyLogits t -> Tensor t -> (CrossEntropyLogits t, Tensor t)
  backward crossEntropy@(CrossEntropyLogits target logits) _ =
    (crossEntropy, -targetMask + softmax)
      where
        targetMask = tensor (shape logits) setClasses
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
