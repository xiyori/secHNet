{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Dataset.Dataset where

import qualified Data.Tensor as T
import Data.ByteString as BS
import GHC.Word (Word8)
import Control.Monad.IO.Class(MonadIO)
import Foreign.C.Types

data CIFAR10 = CIFAR10 {images :: [T.Tensor CFloat], labels :: [CLLong]}

class (Monad m) => Dataset s m t | s -> t where
    (!!) :: s -> Int -> m t
    length :: s -> m Int

partitionBS sz = snd . BS.foldr' (\x (s, l@(block : tail)) ->
    if s < sz then
        (s + 1, (x : block) : tail)
    else  (1, [x]:l)
    ) (0, [[]])

loadImages :: FilePath -> IO [T.Tensor CFloat]
loadImages path = do
    bytes <- BS.readFile path
    let chunks = partitionBS 3073 bytes
    let flat = Prelude.map (Prelude.map ((/ 256) . byteToCFloat) . Prelude.tail) chunks
    let tens = Prelude.map ((`T.view` [3, 32, 32]) . T.fromList) flat
    return tens
    where
        byteToCFloat :: Word8 -> CFloat
        byteToCFloat = fromIntegral

loadLabels :: FilePath -> IO [CLLong]
loadLabels path = do
    bytes <- BS.readFile path
    let chunks = partitionBS 3073 bytes
    let labels = Prelude.map (fromIntegral . Prelude.head) chunks
    return labels

cifar :: IO CIFAR10
cifar = do
        let path = "data/data_batch_"
        let nums = [1..5]
        let ext = ".bin"
        images <- Prelude.concat <$> mapM (loadImages . ((path <>) . (++ ext)) . show) nums
        labs <- Prelude.concat <$> mapM (loadLabels . ((path <>) . (++ ext)) . show) nums
        return $ CIFAR10 images labs


instance (MonadIO m) => Dataset CIFAR10 m (T.Tensor CFloat, CLLong) where
    (!!) :: MonadIO m => CIFAR10 -> Int -> m (T.Tensor CFloat, CLLong)
    (!!) cif i = do
        let image = images cif Prelude.!! i
        let label = labels cif Prelude.!! i
        return (image, label)

    length :: MonadIO m => CIFAR10 -> m Int
    length cif = return $ Prelude.length $ images cif
