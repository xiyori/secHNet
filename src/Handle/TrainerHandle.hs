{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Handle.TrainerHandle where

import Data.Tensor(Tensor(Tensor), zerosLike, shape)
import NN.Optimizer ( Optimizer(step), Momentum)
import Handle.OptimizerHandle ( HasMomentum(..), MomentumHandle(MomentumHandle, getParams, getMomentum) )
import qualified NN.NNDesigner as NN
import Data.IORef(IORef, newIORef, readIORef, writeIORef)
import Data.Layers.Layer (Params(Flat, Nested))
import Control.Monad.IO.Class(MonadIO, liftIO)
import Control.Monad.Reader.Class(MonadReader, asks)
import NN.Autograd as AG
import Data.HashMap (Map)
import qualified Data.Layers.Layer as L
import Foreign.C.Types

data TrainerHandle = TrainerHandle {
    getModel :: NN.NeuralNetwork CFloat,
    getMomentumHandle :: MomentumHandle Params
}

class HasTrainer t where
    trainer :: t -> TrainerHandle

instance HasTrainer TrainerHandle where
    trainer = id

instance HasMomentum TrainerHandle Params where
    momentum = getMomentumHandle . trainer

paramZerosLike :: Params (Tensor CFloat) -> Params (Tensor CFloat)
paramZerosLike (Flat f) = Flat $ map zerosLike f
paramZerosLike (Nested f) = Nested $ map paramZerosLike f

zeroGrad  :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => m ()
zeroGrad = do
    model <- asks (getModel . trainer)
    modelGrads <- NN.getGrads model
    let newModelGrads = paramZerosLike modelGrads
    NN.setGrads model newModelGrads
            

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

forward :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => Map String (Tensor CFloat) -> m (Tensor CFloat)
forward inp = do
    model <- asks (getModel . trainer)
    AG.forward model inp

backward :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => Tensor CFloat -> m (Map String (Maybe(Tensor CFloat)))
backward inp = do
    model <- asks (getModel . trainer)
    AG.backward model inp


newTrainerHandle :: (Monad m, MonadIO m) => NN.NeuralNetwork CFloat -> Momentum -> m TrainerHandle
newTrainerHandle nn m = do
    mh <- liftIO $ newIORef m
    params <- NN.getParams nn
    let zParams = paramZerosLike params
    parh <- liftIO $ newIORef zParams
    pure $ TrainerHandle nn (MomentumHandle mh parh)
    