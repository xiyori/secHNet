{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}

module Instances where

import Control.Monad
import System.Random
import Data.Tensor as T (Tensor, HasDtype, full, randRange, map, (!:), Indexers, Indexer(I, (:.)), Index)
import Test.QuickCheck

import Foreign.C.Types


instance (HasDtype t, UniformRange t, Num t, Eq t) => Arbitrary (Tensor t) where
  arbitrary :: Gen (Tensor t)
  arbitrary = do
    shape <- arbitrary
    arbitraryWithShape shape

instance {-# OVERLAPPING #-} Arbitrary Index where
  arbitrary :: Gen Index
  arbitrary = do
    nDims <- chooseInt (0, 4)
    vectorOf nDims $ chooseEnum (0, 10)

arbitraryWithShape :: (HasDtype t, UniformRange t, Num t, Eq t) =>
  Index -> Gen (Tensor t)
arbitraryWithShape shape = do
  isContiguous <- chooseAny
  if isContiguous then
    arbitraryContiguousWithShape shape
  else do
    isUniform <- chooseAny
    if isUniform then do
      value <- chooseInt (-127, 127)
      if value == 0 then
        return $ full shape (-128)
      else return $ full shape $ fromIntegral value
    else do
      expandedShape <- mapM (
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

arbitrarySlices :: Index -> Index -> Gen Indexers
arbitrarySlices =
  zipWithM (
    \ dim expDim -> do
      start <- chooseEnum (0, expDim - dim)
      isNegative <- chooseAny
      return $
        if isNegative then
          I (-(start + 1)) :. -(start + dim + 1) :. -1
        else
          I start :. start + dim
  )

arbitraryContiguousWithShape :: (HasDtype t, UniformRange t, Num t, Eq t) =>
  Index -> Gen (Tensor t)
arbitraryContiguousWithShape shape = do
  seed <- chooseAny
  case mkStdGen seed of {gen ->
    return
    $ T.map (
      \ value ->
        if value == 0 then
          -128
        else value
    )
    $ fst
    $ randRange shape (-127, 127) gen
  }

arbitraryPairWithShape :: (HasDtype t, UniformRange t, Num t, Eq t) =>
  Index -> Gen (Tensor t, Tensor t)
arbitraryPairWithShape shape = do
  x1 <- arbitraryWithShape shape
  x2 <- arbitraryWithShape shape
  return (x1, x2)

arbitraryBroadcastablePair :: (HasDtype t, UniformRange t, Num t, Eq t) =>
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
    if null shape then
      return (shape, shape)
    else
      case randomR (0, length shape - 1) gen of {(dim1, gen) ->
      case randomR (0, length shape - 1) gen of {(dim2, gen) ->
      case randomR (0, length shape) gen of {(len1, gen) ->
      case randomR (0, length shape) gen of {(len2, gen) ->
        return (drop len1 $ take dim1 shape ++ [1] ++ drop (dim1 + 1) shape,
                drop len2 $ take dim2 shape ++ [1] ++ drop (dim2 + 1) shape)
      }}}}
  }
