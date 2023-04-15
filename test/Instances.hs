{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}

module Instances where

import Data.Vector.Storable (Storable, Vector, (//))
import qualified Data.Vector.Storable as V
import System.Random
import Data.Tensor as T
import Test.QuickCheck

import Foreign.C.Types


instance (Storable t, Random t, Floating t) => Arbitrary (Tensor t) where
  arbitrary :: Gen (Tensor t)
  arbitrary = do
    shape <- arbitrary
    arbitraryWithShape shape

instance Arbitrary (Vector CInt) where
  arbitrary :: Gen (Vector CInt)
  arbitrary = do
    nDims <- chooseInt (0, 4)
    list <- vectorOf nDims $ chooseEnum (0, 10)
    return $ V.fromList list

arbitraryWithShape :: (Storable t, Random t, Floating t) =>
  Vector CInt -> Gen (Tensor t)
arbitraryWithShape shape = do
  seed <- chooseAny
  case mkStdGen seed of {gen ->
    return $ fst $ randn shape gen
  }

arbitraryPairWithShape :: (Storable t, Random t, Floating t) =>
  Vector CInt -> Gen (Tensor t, Tensor t)
arbitraryPairWithShape shape = do
  seed <- chooseAny
  case mkStdGen seed of {gen ->
  case randn shape gen of {(x1, gen) ->
    return (x1, fst $ randn shape gen)
  }}

arbitraryBroadcastablePair :: (Storable t, Random t, Floating t) =>
  Gen (Tensor t, Tensor t)
arbitraryBroadcastablePair = do
  (shape1, shape2) <- arbitraryBroadcastableShapes
  seed <- chooseAny
  case mkStdGen seed of {gen ->
  case randn shape1 gen of {(x1, gen) ->
    return (x1, fst $ randn shape2 gen)
  }}

arbitraryBroadcastableShapes :: Gen (Vector CInt, Vector CInt)
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
