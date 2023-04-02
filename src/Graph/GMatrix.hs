{-# LANGUAGE DeriveFunctor #-}

module Graph.GMatrix where

import Data.Matrix


data GMatrix a
  = -- | Tensor that requres update by the optimizer.
    Parameter (Matrix a)
  | -- | Tensor that is not updated.
    Plain (Matrix a)
  deriving (Functor)
