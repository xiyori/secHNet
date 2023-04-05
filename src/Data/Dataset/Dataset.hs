{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Dataset.Dataset where

import qualified Data.Matrix as M
import Data.ByteString as BS
import Control.Monad.IO.Class(MonadIO)

data CIFAR10 = CIFAR10 {images :: [M.Matrix Int], labels :: [Int]}

class (Monad m) => Dataset s m t | s -> t where
    (!!) :: s -> Int -> m t
    length :: s -> m Int

partitionBS sz = snd . BS.foldr' (\x (s, l@(block : tail)) -> 
    if s < sz then 
        (s + 1, (x : block) : tail) 
    else  (1, [x]:l)
    ) (0, [[]])

batch :: FilePath -> IO [M.Matrix Int]
batch path = do
    bytes <- BS.readFile path
    let chunks = partitionBS 3073 bytes
    let vecs = Prelude.map (Prelude.map fromIntegral . Prelude.tail) chunks
    let mats = Prelude.map (M.fromList 32 32) vecs
    return mats

label :: FilePath -> IO [Int]
label path = do
    bytes <- BS.readFile path
    let chunks = partitionBS 3073 bytes
    let labels = Prelude.map (fromIntegral . Prelude.head) chunks
    return labels

cifar :: IO CIFAR10
cifar = do
        let path = "data/data_batch_"
        let nums = [1..5]
        let ext = ".bin"
        images <- Prelude.concat <$> mapM (batch . ((path <>) . (++ ext)) . show) nums
        labs <- Prelude.concat <$> mapM (label . ((path <>) . (++ ext)) . show) nums
        return $ CIFAR10 images labs

instance (MonadIO m) => Dataset CIFAR10 m (M.Matrix Int, Int) where
    (!!) :: MonadIO m => CIFAR10 -> Int -> m (M.Matrix Int, Int)
    (!!) cif i = do
        let image = images cif Prelude.!! i
        let label = labels cif Prelude.!! i
        return (image, label)

    length :: MonadIO m => CIFAR10 -> m Int
    length cif = return $ Prelude.length $ images cif