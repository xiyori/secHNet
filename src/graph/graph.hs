module Graph.Graph where

import Control.Monad.Free (Free)
import Data.Tensor (Tensor)
import Graph.GTensor (GTensor)

data Operation a s n
  = -- | Input tensor.
    Const (GTensor s n)
  | -- | Sum with a tensor.
    Sum a (GTensor s n)
  | -- | Multiply by a tensor.
    Prod a (GTensor s n)
  | -- | Linear layer with a tensor weight.
    Linear a (GTensor s n)
  | -- | ReLU activation.
    ReLU a
  | -- | Cross-entropy loss with a target tensor.
    CrossEntropyLogits a (GTensor s n)

-- | Computational graph with operations from grammar @f@.
--   Intermediate results are cached for use in backward.
type Graph f s n = Free f (Maybe (Tensor s n))
