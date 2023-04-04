{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
module NN.Autograd where

import Control.Monad (forM_, unless)
import Data.Array.IO(IOArray, newArray, readArray, writeArray)
import qualified Data.HashMap as H
import Data.Matrix
import Control.Monad.IO.Class(MonadIO, liftIO)
import qualified Handle.LayerHandle as L
import NN.NNDesigner
import Data.Bits (Bits(xor))

forward :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> H.Map String (Matrix t) -> m (Matrix t)
forward nn@(NeuralNetwork nodes inputs outputs output) inp = 
    liftIO (newArray (0, length nodes - 1) Nothing) >>= forwardHelper nn inp

forwardHelper :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> H.Map String (Matrix t) -> 
    IOArray Int (Maybe (Matrix t)) -> m (Matrix t)
forwardHelper (NeuralNetwork nodes inputs outputs output) inpValue cache = do
    cached output
    where
        cached i = do
            val <- liftIO $ readArray cache i
            case val of
                Nothing -> do
                    val <- calc (nodes !! i)
                    liftIO $ writeArray cache i (Just val)
                    pure val
                Just res -> pure res

        calc (Input s) = pure $ inpValue H.! s
        calc (x :+: y) = do
            xx <- cached x
            yy <- cached y
            pure $ xx + yy
        calc (LayerNode handle inp) = do
            inpVal <- cached inp
            liftIO $ L.forward handle inpVal
        calc (SubmoduleNode submodule inps) = do
            inpVals <- traverse cached inps
            forward submodule inpVals
        

backward :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> m (H.Map String (Matrix t))
backward nn@(NeuralNetwork nodes inputs outputs output) outValue = do
    used <- liftIO $ newArray (0, length nodes - 1) False
    grad <- liftIO $ newArray (0, length nodes - 1) Nothing
    transposed <- liftIO $ newArray (0, length nodes - 1) []
    backwardHelper nn outValue used grad transposed


backwardHelper :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> 
    IOArray Int Bool -> IOArray Int (Maybe (Matrix t)) -> IOArray Int [Int] -> m (H.Map String (Matrix t))

backwardHelper (NeuralNetwork nodes inputs outputs output) outValue used grad transposed = do
    makeTransposed output
    pure $ H.empty
    where
        makeTransposed :: (Monad m, MonadIO m) => Int -> m()
        makeTransposed i = 
            case nodes !! i of
                (Input _) -> pure ()
                (x :+: y) -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx)
                    yy <- liftIO $ readArray transposed y
                    liftIO $ writeArray transposed y (i : yy)
                (LayerNode _ x) -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx)
                (SubmoduleNode _ map) -> forM_ map (\x -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx) 
                    )
    
        addGrad i val = do
            g <- liftIO $ readArray grad i
            let newG = case g of
                           Nothing -> Just val
                           Just x -> Just (x + val) 
            liftIO $ writeArray grad i newG
         
        relax i = do
            u <- liftIO $ readArray used i -- We haven't calculated grad yet
            unless u $ do
                    out <- readArray transposed i
                    forM_ out relax -- We traverse all our outputs
                    case nodes !! i of 
                        (Input _) -> pure()
                        (x :+: y) -> do
                            g <- liftIO $ readArray grad i
                            let (Just gg) = g
                            addGrad x gg
                            addGrad y gg
                        (LayerNode l inp) -> do
                            g <- liftIO $ readArray grad i
                            let (Just gg) = g
                            inpGrad <- liftIO $ L.backward l gg
                            addGrad inp inpGrad
                        (SubmoduleNode submodule inp) -> do
                            g <- liftIO $ readArray grad i
                            let (Just gg) = g
                            inpGrads <- backward submodule gg
                            let keys = H.keys inp
                            forM_ keys (\key -> addGrad (inp H.! key) (inpGrads H.! key))
