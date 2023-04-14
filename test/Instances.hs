{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}

module Instances where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import System.Random
import Data.Tensor as T
import Test.QuickCheck

import Foreign.C.Types


instance (Storable t, Random t, Floating t) => Arbitrary (Tensor t) where
  arbitrary :: Gen (Tensor t)
  arbitrary = do
    shape <- arbitrary
    seed <- chooseAny
    case mkStdGen seed of {gen ->
      return $ fst $ randn shape gen  
    }

instance Arbitrary (Vector CInt) where
  arbitrary :: Gen (Vector CInt)
  arbitrary = do
    nDims <- chooseInt (0, 4)
    list <- vectorOf nDims $ chooseEnum (0, 10)
    return $ V.fromList list
