module Data.Tensor.Instances where

import Data.Vector.Storable (Storable, Vector)
import Data.Tensor.Definitions
import Data.Tensor.Functional as T


instance NumTensor t => Num (Tensor t) where
  (+) = numTPlus
  (-) = numTMinus
  (*) = numTMult
  abs = numTAbs
  signum = numTSignum
  fromInteger = single . fromInteger

instance FractionalTensor t => Fractional (Tensor t) where
  (/) = fracTDiv
  fromRational = single . fromRational

instance FloatingTensor t => Floating (Tensor t) where
  pi = single pi
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
