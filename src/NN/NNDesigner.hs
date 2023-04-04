{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module NN.NNDesigner where

import Data.Matrix
import Data.HashMap(Map, empty, insert)
import Data.Layers.Layer(Layer)
import Control.Monad(forM)
import Handle.LayerHandle
import Control.Monad.IO.Class(MonadIO, liftIO)
import Control.Monad.Reader(MonadReader)
import Control.Monad.State.Class(MonadState, modify, gets)
import Control.Monad.Trans.State(StateT (runStateT))
import GHC.Read (readField)

data Node t = Input String | LayerNode (LayerHandle t) Int | Int :+: Int | SubmoduleNode (NeuralNetwork t) (Map String Int)
data NeuralNetwork t = NeuralNetwork {
    nodes :: [Node t], 
    inputs :: Map String Int, 
    outputs :: Map String Int, 
    output :: Int}

class (Monad m, MonadState (NeuralNetwork t) m) => MonadNNDesigner m t | m -> t where
    newNode :: Node t -> m Int
    registerOutput :: Int -> String -> m ()

instance (Monad m) => MonadNNDesigner (StateT (NeuralNetwork t) m) t where
    newNode :: (Monad m) => Node t -> StateT (NeuralNetwork t) m Int
    newNode node@(Input name) = do
        modify helper 
        gets ((-) 1 . length . nodes)
        where
          helper (NeuralNetwork nodes inputs outputs output) = 
            NeuralNetwork (node: nodes) (insert name (length nodes) inputs) outputs output
    
    newNode node = do
        modify helper 
        gets ((-) 1 . length. nodes)
        where
          helper (NeuralNetwork nodes inputs outputs output) = 
            NeuralNetwork (node: nodes) inputs outputs output

    registerOutput idx name = modify helper 
        where
          helper (NeuralNetwork nodes inputs outputs output) = 
            NeuralNetwork nodes inputs (insert name idx outputs) output

newLayer :: (Layer l f t, Functor f, MonadIO m, MonadNNDesigner m t) => l -> Int -> m Int
newLayer l inp = do
    handle <- liftIO $ newLayerHandle l
    newNode $ LayerNode handle inp

newSubmodule :: (MonadIO m, MonadNNDesigner m t) => NeuralNetwork t -> Map String Int -> m Int
newSubmodule nn map = do
    newNN <- copyNN nn
    newNode $ SubmoduleNode newNN map

compileNN :: (Monad m, MonadIO m) => StateT (NeuralNetwork t) m (String, Int) -> m (NeuralNetwork t)
compileNN st = do
    ((outName, outId), NeuralNetwork nodes inputs outputs _) <- runStateT st (NeuralNetwork [] empty empty 0)
    pure $ NeuralNetwork nodes inputs (insert outName outId outputs) outId


-- Makes copy of IORefs so that you can use submodule several times
copyNN :: (Monad m, MonadIO m) => NeuralNetwork t -> m (NeuralNetwork t)
copyNN (NeuralNetwork nodes inputs outputs output)= do
    newNodes <- forM nodes helper
    pure $ NeuralNetwork newNodes inputs outputs output
    where 
        helper :: (Monad m, MonadIO m) => Node t -> m (Node t)
        helper (LayerNode handle inp) = do
            handle <- liftIO $ copyLayer handle
            pure $ LayerNode handle inp
        helper (SubmoduleNode nn inp) = do
            newNN <- copyNN nn
            pure $ SubmoduleNode newNN inp
        helper other = pure other


            