module Data.Dataset.Dataloader where

import Data.Dataset.Dataset as DAS
import Data.Conduit
import System.Random
import Data.List (nub)
import Conduit (MonadIO(liftIO), yieldMany, awaitNonNull)
import Control.Monad.Trans.Class (lift)
import Data.Void(Void)
import Control.Monad(forM_)
import Data.Array.IO as MA
randomSample :: (Dataset s m t, MonadIO m, Show t) => s -> ConduitT () t m ()
randomSample ds = do
    gen <- liftIO newStdGen
    len <- lift $ DAS.length ds
    let inds = take (len - 1) $ randomRs (0, len - 1) gen
    forM_ inds $ \i -> do
        sample <- lift $ ds DAS.!! i
        yield sample

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
