{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main where
import Data.Tensor(Tensor, tensor, getElem, flatten, sumAlongDim, eye)
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
import System.Random(newStdGen)
import Data.Index (indexRange0)


mlp :: (MonadNNDesigner m Double, MonadIO m) => m (String, Int)
mlp = do
    rand <- liftIO newStdGen
    inp1 <- newNode $ Input "input"
    lin1 <- newLayer (makeLinear 1024 256 rand) inp1
    relu1 <- newLayer (makeReLU 256) lin1
    lin2 <- newLayer (makeLinear 256 128 rand) relu1
    relu2 <- newLayer (makeReLU 128) lin2
    lin3 <- newLayer (makeLinear 128 10 rand) relu2
    pure ("output", lin3)


flattenAndToDouble :: Tensor Int -> Tensor Double
flattenAndToDouble m = 
    let dbl = fmap fromIntegral m in flatten $ sumAlongDim dbl 0

train :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => ConduitT [(Tensor Int, Int)] Void m ()
train = awaitForever batchStep
    where
        batchStep :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => [(Tensor Int, Int)] -> m()
        batchStep batch = do
            liftIO $ putStrLn "Batch arrived"
            let (feat, labels) = unzip batch
            let dFeat = map flattenAndToDouble feat
            let featT = tensor [length dFeat, 1024] (\idx -> getElem (dFeat !! (head idx)) (tail idx))
            let labelT = tensor [length labels] (\idx -> labels !! (head idx))

            liftIO $ print labelT

            zeroGrad
            
            let mapping = fromList [("input", featT)]
            outputs <- forward mapping
            
            let lossfunc = makeCrossEntropyLogits
            let lossfunc1 = L.setCrossEntropyTarget lossfunc labelT
            let (lossfunc2, loss) = L.forward lossfunc outputs 
            liftIO $ print loss
            let (_, grads) = L.backward lossfunc2 (pure 1)
            backward grads

            optimize

    


main :: IO ()
main = do
    model <- compileNN mlp
    handle <- newTrainerHandle model (Momentum 0.9 0.01)
    ds <- DS.cifar
    putStrLn "Starting train"
    let loader = randomSample ds
    let pipeline = loader .| (toBatch 32) .| train
    runReaderT (runConduit pipeline) handle
