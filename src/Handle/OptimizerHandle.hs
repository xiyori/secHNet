{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleInstances #-}

module Handle.OptimizerHandle where

import Data.Tensor (Tensor)
import Data.Layers.Layer (Params)
import NN.Optimizer(Optimizer, Momentum (Momentum))
import Data.IORef(IORef, newIORef)
import Control.Monad.IO.Class(MonadIO, liftIO)
import Foreign.C.Types


data MomentumHandle f = MomentumHandle { 
    getMomentum :: IORef Momentum, 
    getParams :: IORef (f (Tensor CFloat))
}

class HasMomentum m f | m -> f where
    momentum :: m -> MomentumHandle f

instance HasMomentum (MomentumHandle f) f where
    momentum = id



