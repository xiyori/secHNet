module Data.Tensor.Instances where

import Data.Vector.Storable (Storable, Vector)
import Data.Tensor.Definitions
import Data.Tensor.Functional as T


instance Storable t => Eq (Tensor t) where
  (==) = tensorEqual

instance NumTensor t => Num (Tensor t) where
  (+) = numTAdd
  (-) = numTSub
  (*) = numTMult
  negate = numTNegate
  abs = numTAbs
  signum = numTSignum
  fromInteger = scalar . fromInteger

instance FractionalTensor t => Fractional (Tensor t) where
  (/) = fracTDiv
  fromRational = scalar . fromRational

instance FloatingTensor t => Floating (Tensor t) where
  pi = scalar pi
  exp = floatTExp
  log = floatTLog
  sin = floatTSin
  cos = floatTCos
  asin = floatTAsin
  acos = floatTAcos
  atan = floatTAtan
  sinh = floatTSinh
  cosh = floatTCosh
  asinh = floatTAsinh
  acosh = floatTAcosh
  atanh = floatTAtanh
