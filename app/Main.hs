{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main where
import Data.Matrix(zero, Matrix, setSize, identity)
import NN.NNDesigner(MonadNNDesigner, newNode, newLayer, Node((:+:), Input), compileNN)
import Control.Monad.IO.Class(MonadIO, liftIO)
import Data.Layers.Layer(makeLinear, makeReLU, makeCrossEntropyLogits)
import qualified Data.Layers.Layer as L
import Data.HashMap (fromList)
import qualified Data.Dataset.Dataset as DS
import Data.Dataset.Dataloader
import Handle.TrainerHandle
import Control.Monad.IO.Class(MonadIO, liftIO)
import Control.Monad.Reader.Class (MonadReader)
import Control.Monad.Trans.Reader(runReaderT)
import Conduit(ConduitT, awaitForever, runConduit, (.|))
import Control.Monad (forever, forM_)
import NN.Optimizer(Momentum(Momentum))
import Data.Void(Void)


mlp :: (MonadNNDesigner m Double, MonadIO m) => m (String, Int)
mlp = do
    inp1 <- newNode $ Input "input"
    lin1 <- newLayer (makeLinear 1024 1024) inp1
    relu1 <- newLayer (makeReLU 1024) lin1
    lin2 <- newLayer (makeLinear 1024 256) relu1
    relu2 <- newLayer (makeReLU 256) lin2
    lin3 <- newLayer (makeLinear 256 10) relu2
    pure ("output", lin3)

flattenAndToDouble :: Matrix Int -> Matrix Double
flattenAndToDouble m = setSize 0 1024 1 (fmap fromIntegral m) 

train :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => ConduitT [(Matrix Int, Int)] Void m ()
train = awaitForever batchStep
    where
        batchStep :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => [(Matrix Int, Int)] -> m()
        batchStep batch = do
            liftIO $ putStrLn "Batch arrived"
            let (feat, labels) = unzip batch
            let dFeat = map flattenAndToDouble feat
            zeroGrad
            forM_ (zip dFeat labels) (uncurry trainStep)
            optimize

        trainStep :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => (Matrix Double) -> Int -> m()
        trainStep m l = do
            let mapping = fromList [("input", m)]
            outputs <- forward mapping
            liftIO $ print l
            let lossfunc = makeCrossEntropyLogits l 10
            let (lossfunc1, loss) = L.forward lossfunc outputs 
            liftIO $ print loss
            let (_, grads) = L.backward lossfunc1 (identity 1)
            backward grads
            pure ()

    


main :: IO ()
main = do
    model <- compileNN mlp
    handle <- newTrainerHandle model (Momentum 0.9 0.01)
    ds <- DS.cifar
    putStrLn "Starting train"
    let loader = randomSample ds
    let pipeline = loader .| (toBatch 32) .| train
    runReaderT (runConduit pipeline) handle
