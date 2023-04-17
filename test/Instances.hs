{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}

module Instances where

import Control.Monad
import Data.Vector.Storable (Storable, Vector, (//))
import qualified Data.Vector.Storable as V
import System.Random
import Data.Tensor as T
import Test.QuickCheck

import Foreign.C.Types


instance (HasDtype t, Random t, Floating t) => Arbitrary (Tensor t) where
  arbitrary :: Gen (Tensor t)
  arbitrary = do
    shape <- arbitrary
    arbitraryWithShape shape

instance Arbitrary Index where
  arbitrary :: Gen Index
  arbitrary = do
    nDims <- chooseInt (0, 4)
    list <- vectorOf nDims $ chooseEnum (0, 10)
    return $ V.fromList list

arbitraryWithShape :: (HasDtype t, Random t, Floating t) =>
  Index -> Gen (Tensor t)
arbitraryWithShape shape = do
  isContiguous <- chooseAny
  if isContiguous then
    arbitraryContiguousWithShape shape
  else do
    expandedShape <- V.mapM (
      \ dim -> do
        addDim <- chooseEnum (0, 10)
        return $ dim + addDim
      ) shape
    slices <- arbitrarySlices shape expandedShape
    x <- arbitraryContiguousWithShape expandedShape
    return $ x !: slices

-- arbitraryWithShape :: (HasDtype t, Random t, Floating t) =>
--   Vector CInt -> Gen (Tensor t)
-- arbitraryWithShape = arbitraryContiguousWithShape

arbitrarySlices :: Index -> Index -> Gen Slices
arbitrarySlices shape expandedShape =
  zipWithM (
    \ dim expDim -> do
      start <- chooseEnum (0, expDim - dim)
      isNegative <- chooseAny
      return $
        if isNegative then
          -(start + 1) :. -(start + dim + 1) :| -1
        else
          start :. start + dim
  ) (V.toList $ V.map fromIntegral shape)
    (V.toList $ V.map fromIntegral expandedShape)

arbitraryContiguousWithShape :: (HasDtype t, Random t, Floating t) =>
  Index -> Gen (Tensor t)
arbitraryContiguousWithShape shape = do
  seed <- chooseAny
  case mkStdGen seed of {gen ->
    return $ fst $ randn shape gen
  }

arbitraryPairWithShape :: (HasDtype t, Random t, Floating t) =>
  Index -> Gen (Tensor t, Tensor t)
arbitraryPairWithShape shape = do
  x1 <- arbitraryWithShape shape
  x2 <- arbitraryWithShape shape
  return (x1, x2)

arbitraryBroadcastablePair :: (HasDtype t, Random t, Floating t) =>
  Gen (Tensor t, Tensor t)
arbitraryBroadcastablePair = do
  (shape1, shape2) <- arbitraryBroadcastableShapes
  x1 <- arbitraryWithShape shape1
  x2 <- arbitraryWithShape shape2
  return (x1, x2)

arbitraryBroadcastableShapes :: Gen (Index, Index)
arbitraryBroadcastableShapes = do
  shape <- arbitrary
  seed <- chooseAny
  case mkStdGen seed of {gen ->
    if V.length shape == 0 then
      return (shape, shape)
    else
      case randomR (0, V.length shape - 1) gen of {(dim1, gen) ->
      case randomR (0, V.length shape - 1) gen of {(dim2, gen) ->
      case randomR (0, V.length shape) gen of {(len1, gen) ->
      case randomR (0, V.length shape) gen of {(len2, gen) ->
        return (V.drop len1 $ shape // [(dim1, 1)],
                V.drop len2 $ shape // [(dim2, 1)])
      }}}}
  }
