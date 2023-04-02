module Optimizer.SGD where

import Data.Matrix
import Layers.Layer
import Gradient.Gradient
import Optimizer.Optimizer

data SGD = SGD {learningRate :: Float}

instance Optimizer SGD where
    stepParam (SGD eta) param grad = param - grad * eta