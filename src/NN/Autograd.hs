{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
module NN.Autograd where

import Control.Monad (forM_)
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
            pure $ elementwise (+) xx yy
        calc (LayerNode handle inp) = do
            inpVal <- cached inp
            liftIO $ L.forward handle inpVal
        calc (SubmoduleNode submodule inps) = do
            inpVals <- traverse cached inps
            forward submodule inpVals
        

-- backward :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> m (H.Map String (Matrix t))
-- backward nn@(NeuralNetwork nodes inputs outputs output) outValue = do
--     cache <- liftIO $ newArray (0, length nodes - 1) Nothing
--     transposed <- liftIO $ newArray (0, length nodes - 1) []
--     backwardHelper nn outValue cache transposed


-- backwardHelper :: (Monad m, MonadIO m, Floating t) => NeuralNetwork t -> Matrix t -> 
--     IOArray Int (Maybe (Matrix t)) -> IOArray Int [Int] -> m (H.Map String (Matrix t))

-- backwardHelper (NeuralNetwork nodes inputs outputs output) outValue cache transposed = 
--     makeTransposed output
--     error "TODO"
--     where
--         makeTransposed :: (Monad m, MonadIO m) => Int -> m()
--         makeTransposed i = 
--             case nodes !! i of
--                 (Input _) -> pure ()
--                 (x :+: y) -> do
--                     xx <- liftIO $ readArray transposed x
--                     liftIO $ writeArray transposed x (i : xx)
--                     yy <- liftIO $ readArray transposed y
--                     liftIO $ writeArray transposed y (i : yy)
--                 (LayerNode _ x) -> do
--                     xx <- liftIO $ readArray transposed x
--                     liftIO $ writeArray transposed x (i : xx)
--                 (SubmoduleNode _ map) -> forM_ map (\x -> do
--                     xx <- liftIO $ readArray transposed x
--                     liftIO $ writeArray transposed x (i : xx) 
--                     )
    
--         cached i = do
--             val <- liftIO $ readArray cache i -- We haven't calculated grad yet
--             case val of
--                 Nothing -> do
--                     out <- readArray transposed i
--                     forM_ out cached -- We traverse all our outputs
--                     calc (nodes !! i)
--                 Just res -> pure ()

--         calc (Input s) = pure $ inpValue H.! s
--         calc (x :+: y) = do
--             xx <- cached x
--             yy <- cached y
--             pure $ elementwise (+) xx yy
--         calc (LayerNode handle inp) = do
--             inpVal <- cached inp
--             liftIO $ L.forward handle inpVal
--         calc (SubmoduleNode submodule inps) = do
--             inpVals <- traverse cached inps
--             forward submodule inpVals