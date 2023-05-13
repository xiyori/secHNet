{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FunctionalDependencies #-}

module Data.Layers.Layer where

import Control.Applicative
import System.Random
import Data.Tensor (HasDtype, Tensor, HasArange(arange), tensor, zeros, randn,
                    (@), (!), Indexer (T), (!:), (!=), shape, sumAlongDims, sumAlongDim, relu, astype, transpose)
import qualified Data.Tensor as T

import Foreign.C.Types

data Params t = Flat [t] | Nested [Params t]

instance Functor Params where
  fmap :: (a -> b) -> Params a -> Params b
  fmap f (Flat x) = Flat $ map f x
  fmap f (Nested x) = Nested $ map (fmap f) x

instance Applicative Params where
    pure :: a -> Params a
    pure x = Flat [x]

    liftA2 :: (a -> b -> c) -> Params a -> Params b -> Params c
    liftA2 f (Flat x) (Flat y) = Flat $ zipWith f x y
    liftA2 f (Nested x) (Nested y) = Nested $ zipWith (liftA2 f) x y
    liftA2 _ _ _ = error "Incorrect shape"

instance (Show t) => Show (Params t) where
  show :: Show t => Params t -> String
  show (Flat x) = show x
  show (Nested x) = show x


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
  -- | [out_channels, in_channels, kernel, kernel]
  conv2dWeight :: Tensor t,
  -- | [out_channels, in_channels, kernel, kernel]
  conv2dGradWeight :: Tensor t,
  -- | [out_channels]
  conv2dBias :: Tensor t,
  -- | [out_channels]
  conv2dGradBias :: Tensor t,
  -- | [batch, in_channels]
  conv2dInput :: Tensor t,
  _stride :: Int,
  _padding :: Int
}

-- | In channels -> out channels -> kernel_size -> stride -> padding ->
-- | random gen
makeConv2d :: (RandomGen g) =>
  Int -> Int -> Int -> Int -> Int -> g -> Conv2d CFloat
makeConv2d inChannels outChannels kernel stride padding gen =
  Conv2d (weight * std) (zeros shape)
    (bias * std) (zeros [outChannels]) 0
    stride padding
  where
    std = 1 / sqrt (fromIntegral $ inChannels * kernel * kernel)
    (weight, newGen1) = randn shape gen
    (bias, newGen2) = randn [outChannels] newGen1
    shape = [inChannels, outChannels, kernel, kernel]

instance Num t => Layer (Conv2d t) t where
  forward :: Conv2d t -> Tensor t -> (Conv2d t, Tensor t)
  forward = error "TODO"

  backward :: Conv2d t -> Tensor t -> (Conv2d t, Tensor t)
  backward = error "TODO"

  getParams x = Flat [conv2dWeight x, conv2dBias x]
  getGrads x = Flat [conv2dGradWeight x, conv2dGradBias x]
  setParams (Conv2d _ gradWeight _ gradBias input s p) (Flat l) =
    Conv2d (head l) gradWeight (l !! 1) gradBias input s p
  setGrads (Conv2d weight _ bias _ input s p) (Flat l) =
    Conv2d weight (head l) bias (l !! 1) input s p


data Linear t = Linear {
  -- | [out_channels, in_channels]
  linearWeight :: Tensor t,
  -- | [out_channels, in_channels]
  linearGradWeight :: Tensor t,
  -- | [out_channels]
  linearBias :: Tensor t,
  -- | [out_channels]
  linearGradBias :: Tensor t,
  -- | [batch, in_channels]
  linearInput :: Tensor t
}

-- | In channels -> out channels -> random gen
makeLinear :: (RandomGen g) => Int -> Int -> g -> Linear CFloat
makeLinear inChannels outChannels gen =
  Linear (weight * std) (zeros shape)
    (bias * std) (zeros [outChannels]) 0
  where
    std = 1 / sqrt (fromIntegral inChannels)
    (weight, newGen1) = randn shape gen
    (bias, newGen2) = randn [outChannels] newGen1
    shape = [inChannels, outChannels]

instance (HasDtype t, Floating t) => Layer (Linear t) t where
  forward :: Linear t -> Tensor t -> (Linear t, Tensor t)
  forward (Linear weight gradWeight bias gradBias _) input =
    (Linear weight gradWeight bias gradBias input,
    input @ weight + bias)

  backward :: Linear t -> Tensor t -> (Linear t, Tensor t)
  backward (Linear weight gradWeight bias gradBias input) grad_output =
    (Linear weight (gradWeight + newGradWeight)
    bias (gradBias + newGradBias) input, grad_input)
      where
        grad_input = grad_output @ transpose weight
        newGradWeight = transpose input @ grad_output
        newGradBias = sumAlongDim grad_output 0

  getParams x = Flat [linearWeight x, linearBias x]
  getGrads x = Flat [linearGradWeight x, linearGradBias x]
  setParams (Linear _ gradWeight _ gradBias input) (Flat l) =
    Linear (head l) gradWeight (l !! 1) gradBias input
  setGrads (Linear weight _ bias _ input) (Flat l) =
    Linear weight (head l) bias (l !! 1) input


instance HasDtype t => Show (Linear t) where
  show (Linear weight _ _ _ _) = "Linear (" ++ show (shape weight !! 1) ++ "," ++ show (head $ shape weight) ++ ")"

newtype ReLU t = ReLU {
  reluInput :: Tensor t
}

makeReLU :: Int -> ReLU CFloat
makeReLU i = ReLU (zeros [i, 1])

instance (HasDtype t, Ord t, Num t) => Layer (ReLU t) t where
  forward :: ReLU t -> Tensor t -> (ReLU t, Tensor t)
  forward _ input = (ReLU input, relu input)

  backward :: ReLU t -> Tensor t -> (ReLU t, Tensor t)
  backward relu@(ReLU input) grad_output =
    (relu, grad_output * astype (input T.> 0))

  getParams _ = Flat []
  getGrads _ = Flat []
  setParams = const
  setGrads = const

instance HasDtype t => Show (ReLU t) where
  show (ReLU inp) = "ReLU (" ++ show (head (shape inp)) ++ ")"



-- | Class index from 1 to @n_classes@
data CrossEntropyLogits t = CrossEntropyLogits {
  -- | [batch]
  сrossEntropyTarget :: Tensor CLLong,
  -- | [batch, n_classes]
  сrossEntropyInput :: Tensor t
}

makeCrossEntropyLogits :: CrossEntropyLogits CFloat
makeCrossEntropyLogits = CrossEntropyLogits 0 0

instance (HasDtype t, Floating t) => Layer (CrossEntropyLogits t) t where
  forward :: CrossEntropyLogits t -> Tensor t -> (CrossEntropyLogits t, Tensor t)
  forward (CrossEntropyLogits target _) logits =
    (CrossEntropyLogits target logits,
    -logitsForAnswers + log (sumAlongDim (exp logits) (-1)))
    where
      batch = fromIntegral $ head $ shape logits
      logitsForAnswers = logits !: [T (arange 0 batch 1), T target]

  backward :: CrossEntropyLogits t -> Tensor t -> (CrossEntropyLogits t, Tensor t)
  backward crossEntropy@(CrossEntropyLogits target logits) _ =
    (crossEntropy, -targetMask + softmax)
      where
        batch = fromIntegral $ head $ shape logits
        targetMask = zeros (shape logits) != ([T (arange 0 batch 1), T target], 1)
        softmax = expLogits / sumAlongDims expLogits [-1] True
        expLogits = exp logits

  getParams x = Flat []
  getGrads x = Flat []
  setParams = const
  setGrads = const


setCrossEntropyTarget :: CrossEntropyLogits t -> Tensor CLLong -> CrossEntropyLogits t
setCrossEntropyTarget (CrossEntropyLogits target input) newTarget =
  CrossEntropyLogits newTarget input

instance (HasDtype t) => Show (CrossEntropyLogits t) where
  show (CrossEntropyLogits _ inp) = "CrossEntropyLogits (" ++ show (head (shape inp)) ++ ")"
