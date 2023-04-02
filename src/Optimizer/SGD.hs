{-# LANGUAGE InstanceSigs #-}

module Optimizer.SGD where

import Data.Matrix
import Layers.Layer
import Gradient.Gradient
import Optimizer.Optimizer

newtype SGD = SGD {learningRate :: Float}

instance Optimizer SGD where
  stepParam :: SGD -> Matrix Float -> Matrix Float -> Matrix Float
  stepParam (SGD eta) param grad = param - scaleMatrix eta grad