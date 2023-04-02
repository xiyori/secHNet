{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GADTs #-}

module Graph.Graph where

import Control.Monad.Free (Free)
import Data.Matrix (Matrix)
import Graph.GMatrix
import Layers.Layer

data Operation t a where
   -- | Input tensor.
    Operation :: (Layer l t) => l -> a -> Operation t a

-- | Computational graph with operations from grammar @f@.
--   Intermediate results are cached for use in backward.
type Graph f t = Free f (Matrix t)





