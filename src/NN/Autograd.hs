module NN.Autograd where

import Data.HashMap(Map)
import Data.Matrix
import Control.Monad.IO.Class(MonadIO)
import NN.NNDesigner(NeuralNetwork)



forward :: (Monad m, MonadIO m) => NeuralNetwork t -> Map String (Matrix t) -> m ()
forward nn inp = error "TODO"
