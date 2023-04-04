module Data.Dataset.Dataloader where

import Data.Dataset.Dataset as DAS
import Data.Conduit
import System.Random
import Data.List (nub)
import Conduit (MonadIO(liftIO), yieldMany, awaitNonNull)
import Control.Monad.Trans.Class (lift)

randomSample :: (Dataset s m t, MonadIO m) => s -> ConduitT () t m ()
randomSample ds = do
    gen <- liftIO newStdGen
    len <- lift $ DAS.length ds
    let inds = take len $ nub $ randomRs (0, len - 1) gen
    rand <- lift $ mapM (ds DAS.!!) inds
    yieldMany rand

toBatch :: Monad m => Int -> ConduitT t [t] m ()
toBatch size = helper []
    where 
        helper batch = do
            val <- await
            case val of
                Nothing -> return ()
                Just val' -> do
                    let batch' = val' : batch
                    if Prelude.length batch' < size
                        then helper batch'
                        else yield batch' >> helper []