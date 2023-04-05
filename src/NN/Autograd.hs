{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE FlexibleContexts #-}
module NN.Autograd where

import Data.Foldable(foldrM)
import Control.Monad (forM_, unless)
import Data.Array.IO(IOArray, newArray, readArray, writeArray, getElems)
import qualified Data.HashMap as H
import Data.Matrix
import Control.Monad.IO.Class(MonadIO, liftIO)
import qualified Handle.LayerHandle as L
import NN.NNDesigner
    ( NeuralNetwork(NeuralNetwork),
      Node(SubmoduleNode, Input, (:+:), LayerNode) )
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
        

backward :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> m (H.Map String (Maybe(Matrix t)))
backward nn@(NeuralNetwork nodes inputs outputs output) outValue = do
    used <- liftIO $ newArray (0, length nodes - 1) False
    grad <- liftIO $ newArray (0, length nodes - 1) Nothing
    transposed <- liftIO $ newArray (0, length nodes - 1) []
    backwardHelper nn outValue used grad transposed


backwardHelper :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> 
    IOArray Int Bool -> IOArray Int (Maybe (Matrix t)) -> IOArray Int [Int] -> m (H.Map String (Maybe (Matrix t)))

backwardHelper (NeuralNetwork nodes inputs outputs output) outValue used grad transposed = do
    makeTransposed output
    liftIO $ writeArray grad output (Just outValue)
    forM_ inputs relax
    outGrads <- foldrM (\(k, v) l -> do
        g <- liftIO $ readArray grad v
        pure ((k, g) : l)) [] (H.toList inputs)
    pure $ H.fromList outGrads
    where
        makeTransposed :: (Monad m, MonadIO m) => Int -> m()
        makeTransposed i = 
            case nodes !! i of
                (Input _) -> pure ()
                (x :+: y) -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx)
                    makeTransposed x
                    yy <- liftIO $ readArray transposed y
                    liftIO $ writeArray transposed y (i : yy)
                    makeTransposed y
                (LayerNode _ x) -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx)
                    makeTransposed x
                (SubmoduleNode _ map) -> forM_ map (\x -> do
                    xx <- liftIO $ readArray transposed x
                    liftIO $ writeArray transposed x (i : xx) 
                    makeTransposed x
                    )
    
        addGrad i val = do
            g <- liftIO $ readArray grad i
            let newG = case g of
                           Nothing -> Just val
                           Just x -> Just (x + val) 
            liftIO $ writeArray grad i newG

        relax :: (Monad m, MonadIO m) => Int -> m()
        relax i = do
            u <- liftIO $ readArray used i -- We haven't calculated grad yet
            unless u $ do
                    out <- liftIO $ readArray transposed i
                    forM_ out relax -- We traverse all our outputs
                    case nodes !! i of 
                        (Input _) -> pure()
                        (x :+: y) -> do
                            g <- liftIO $ readArray grad i
                            case g of 
                                Just gg -> do
                                    addGrad x gg
                                    addGrad y gg
                                Nothing -> pure ()
                        (LayerNode l inp) -> do
                            g <- liftIO $ readArray grad i
                            case g of
                                Just gg -> do
                                    inpGrad <- liftIO $ L.backward l gg
                                    addGrad inp inpGrad
                                Nothing -> pure ()
                        (SubmoduleNode submodule inp) -> do
                            g <- liftIO $ readArray grad i
                            case g of
                                Just gg -> do
                                    inpGrads <- backward submodule gg
                                    let keys = H.keys inp
                                    forM_ keys (\key -> case (inpGrads H.! key) of 
                                                    Just inpGrad -> addGrad (inp H.! key) inpGrad
                                                    Nothing -> pure())
                                Nothing -> pure ()
