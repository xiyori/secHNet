{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}

module Handle.OptimizerHandle where

import Data.Matrix (Matrix)
import Data.Layers.Layer (Params)
import NN.Optimizer(Optimizer, Momentum (Momentum))
import Data.IORef(IORef, newIORef)
import Control.Monad.IO.Class(MonadIO, liftIO)


data MomentumHandle f = MomentumHandle { 
    getMomentum :: IORef Momentum, 
    getParams :: IORef (f (Matrix Double))
}

class HasMomentum m f | m -> f where
    momentum :: m -> MomentumHandle f

instance HasMomentum (MomentumHandle f) f where
    momentum = id



