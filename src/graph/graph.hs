{-# LANGUAGE DeriveFunctor #-}

module Graph.Graph where

import Control.Monad.Free (Free)
import Data.Matrix(Matrix)
import Graph.GMatrix

data Operation t a = 
   -- | Input tensor.
    Const (GMatrix t)
   -- | Sum with a tensor.
    | Sum a a
   -- | Multiply by a tensor.
    | Prod a a
   -- | Linear layer with a tensor weight.
    | Linear a (GMatrix t)
   -- | ReLU activation.
    | ReLU a
   -- | Cross-entropy loss with a target tensor.
    | CrossEntropyLogits a (GMatrix t) deriving (Functor)
-- | Computational graph with operations from grammar @f@.
--   Intermediate results are cached for use in backward.
type Graph f t = Free f (Matrix t)

data OperationWithCache t a = OperationWithForward (Operation t a) (Matrix t) (Matrix t )deriving (Functor)



