{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main where
import Data.Tensor as T
import NN.NNDesigner(MonadNNDesigner, newNode, newLayer, Node((:+:), Input), compileNN, getParams, getGrads)
import Control.Monad.IO.Class(MonadIO, liftIO)
import Data.Layers.Layer(makeLinear, makeReLU, makeCrossEntropyLogits)
import qualified Data.Layers.Layer as L
import qualified Data.HashMap as HM
import qualified Data.Dataset.Dataset as DS
import Data.Dataset.Dataloader
import Handle.TrainerHandle
import Control.Monad.IO.Class(MonadIO, liftIO)
import Control.Monad.Reader.Class (MonadReader, asks)
import Control.Monad.Trans.Reader(runReaderT)
import Conduit(ConduitT, awaitForever, runConduit, (.|))
import Control.Monad (forever, forM_)
import NN.Optimizer(Momentum(Momentum))
import Data.Void(Void)
import System.Random(newStdGen)
import Foreign.C (CFloat, CLLong)


mlp :: (MonadNNDesigner m CFloat, MonadIO m) => m (String, Int)
mlp = do
    rand <- liftIO newStdGen
    inp1 <- newNode $ Input "input"
    lin1 <- newLayer (makeLinear 1024 256 rand) inp1
    relu1 <- newLayer (makeReLU 1024) lin1
    lin2 <- newLayer (makeLinear 256 128 rand) relu1
    relu2 <- newLayer (makeReLU 128) lin2
    lin3 <- newLayer (makeLinear 128 10 rand) relu2
    pure ("output", lin3)


train :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => ConduitT [(Tensor CFloat, CLLong)] Void m ()
train = awaitForever batchStep
    where
        batchStep :: (Monad m, MonadIO m, MonadReader e m, HasTrainer e) => [(Tensor CFloat, CLLong)] -> m()
        batchStep batch = do
            liftIO $ putStrLn "Batch arrived"
            let (feat, labels) = unzip batch
            let flatFeat = Prelude.map ((`T.view` [1024]) . (`T.sumAlongDim` 0)) feat 
            let batchFeat = T.tensor [length flatFeat, 1024] (\(h: t) -> (flatFeat Prelude.!! h) T.! t) -- slow, need faster concat
            let batchLbl = fromList labels

            zeroGrad
            
            let mapping = HM.fromList [("input", batchFeat)]
            outputs <- forward mapping
            
            -- liftIO $ print "Labels:"
            -- liftIO $ print batchLbl

            -- liftIO $ print "Features:"
            -- liftIO $ print batchFeat
            -- model <- asks (getModel . trainer)
            -- params <- getParams model
            -- liftIO $ print "Params:"
            -- liftIO $ print params

            -- liftIO $ print "Outputs:"
            -- liftIO $ print outputs

            let lossfunc = makeCrossEntropyLogits
            let lossfunc1 = L.setCrossEntropyTarget lossfunc batchLbl
            let (lossfunc2, loss) = L.forward lossfunc1 outputs 
            -- liftIO $ print "Loss:"
            liftIO $ print $ T.mean loss

            let (_, grads) = L.backward lossfunc2 (T.scalar (1 / fromIntegral (length labels)))

            -- liftIO $ print "Output_grads:"
            -- liftIO $ print grads

            backward grads
            
            -- mgrads <- getGrads model
            -- liftIO $ print "Grads:"
            -- liftIO $ print mgrads

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
