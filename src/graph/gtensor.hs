{-# LANGUAGE DeriveFunctor #-}

module Graph.GTensor where

import Data.Tensor (Tensor)

data GTensor s n
  = -- | Tensor that requres update by the optimizer.
    Parameter (Tensor s n)
  | -- | Tensor is not updated.
    Plain (Tensor s n)
  deriving (Functor)
