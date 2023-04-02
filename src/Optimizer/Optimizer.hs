{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Optimizer.Optimizer where

import Data.Matrix
import Layers.Layer
import Gradient.Gradient

zeroMatrix :: Matrix Float -> Matrix Float
zeroMatrix mat = zero (nrows mat) (ncols mat)

zeroGradOne :: (HasParams Float a) => a -> a
zeroGradOne x = 
    setGrads x $ map zeroMatrix $ getGrads x

zeroGrad :: (Network Float) -> (Network Float)
zeroGrad (Network layers) = 
    Network $ map (\(LayerT l) -> LayerT (zeroGradOne l)) layers
    

class Optimizer a where
    stepParam :: a -> Matrix Float -> Matrix Float -> Matrix Float

stepOne :: (HasParams Float a, Optimizer o) => o -> a -> a
stepOne opt x = setParams x $ map (\(grad, param) -> stepParam opt grad param) $ zip (getGrads x) (getParams x)

step :: (Optimizer o) => o -> (Network Float) -> (Network Float)
step opt (Network layers) = Network $ map (\(LayerT l) -> LayerT (stepOne opt l)) layers