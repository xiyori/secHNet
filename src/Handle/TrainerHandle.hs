{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Handle.TrainerHandle where

import Data.Matrix
import NN.Optimizer ( Optimizer(step) )
import Handle.OptimizerHandle ( HasMomentum(..), MomentumHandle(getParams, getMomentum) )
import qualified NN.NNDesigner as NN
import Data.IORef(IORef, newIORef, readIORef, writeIORef)
import Data.Layers.Layer (Params(Flat, Nested))
import Control.Monad.IO.Class(MonadIO, liftIO)
import Control.Monad.Reader.Class(MonadReader, asks)

data TrainerHandle = TrainerHandle {
    getModel :: NN.NeuralNetwork Double,
    getMomentumHandle :: MomentumHandle Params
}

class HasTrainer t where
    trainer :: t -> TrainerHandle

instance HasTrainer TrainerHandle where
    trainer = id

instance HasMomentum TrainerHandle Params where
    momentum = getMomentumHandle . trainer


zeroGrad  :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => m ()
zeroGrad = do
    model <- asks (getModel . trainer)
    modelGrads <- NN.getGrads model
    let newModelGrads = zeroLike modelGrads
    NN.setGrads model newModelGrads
        where 
            zeroLike :: Params (Matrix Double) -> Params (Matrix Double)
            zeroLike (Flat f) = Flat $ map (\m -> zero (nrows m) (ncols m)) f
            zeroLike (Nested f) = Nested $ map zeroLike f


optimize :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => m ()
optimize = do
    model <- asks (getModel . trainer)
    modelParams <- NN.getParams model
    modelGrads <- NN.getGrads model
    optHandle <- asks (momentum . trainer)
    optim <- liftIO $ readIORef (getMomentum optHandle)
    optParams <- liftIO $ readIORef (getParams optHandle)
    let (newOpt, newModelParams, newOptParams) = step optim modelParams modelGrads optParams 
    NN.setParams model newModelParams
    liftIO $ writeIORef (getMomentum optHandle) newOpt
    liftIO $ writeIORef (getParams optHandle) newOptParams
    