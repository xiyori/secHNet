{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main where
import NN.NNDesigner(MonadNNDesigner, newNode, newLayer, Node((:+:), Input), compileNN)
import NN.Autograd(forward, backward)
import Control.Monad.IO.Class(MonadIO, liftIO)
import Data.Layers.Layer(makeLinear, makeReLU, makeCrossEntropyLogits)
import Data.HashMap (fromList)
import Data.Matrix(zero)


mlp :: (MonadNNDesigner m Double, MonadIO m) => m (String, Int)
mlp = do
    inp1 <- newNode $ Input "inp"
    inp2 <- newNode $ Input "inp2"
    s <- newNode $ inp1 :+: inp2
    lin1 <- newLayer (makeLinear 5 5) s
    relu1 <- newLayer (makeReLU 5) lin1
    lin2 <- newLayer (makeLinear 5 3) relu1
    relu2 <- newLayer (makeCrossEntropyLogits 0 3) lin2
    pure ("output", relu2)
    

main :: IO ()
main = do
    compiled <- compileNN mlp
    let mapping = fromList [("inp", (zero @Double 5 1)), ("inp2", (zero @Double 5 1))]
    print compiled
    fw <- forward compiled mapping
    grads <- backward compiled fw
    print fw
    print grads
    putStrLn "Hello pain"
