{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main where
import NN.NNDesigner(MonadNNDesigner, newNode, newLayer, Node((:+:), Input), compileNN)
import NN.Autograd(forward, backward)
import Control.Monad.IO.Class(MonadIO, liftIO)
import Data.Layers.Layer(makeLinear, makeReLU, makeCrossEntropyLogits)
import Data.HashMap (singleton)
import Data.Matrix


mlp :: (MonadNNDesigner m Double, MonadIO m) => m (String, Int)
mlp = do
    inp <- newNode $ Input "inp"
    inp2 <- newNode $ Input "inp2"
    res <- newNode (inp :+: inp2)
    lin1 <- newLayer (makeLinear 5 5) res
    relu1 <- newLayer (makeCrossEntropyLogits 5 5) lin1
    lin2 <- newLayer (makeLinear 5 1) relu1
    pure ("output", relu1)
    

main :: IO ()
main = do
    compiled <- compileNN mlp
    let mapping = singleton "input" (zero @Double 5 1)
    fw <- forward compiled mapping
    print fw
    putStrLn "Hello pain"
