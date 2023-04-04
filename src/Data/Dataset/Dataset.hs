{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Dataset.Dataset where

import qualified Data.Matrix as M
import Data.ByteString as BS

data CIFAR10 = CIFAR10 {images :: [M.Matrix Int], labels :: [Int]}

class (Monad m) => Dataset s m t | s -> t where
    (!!) :: s -> Int -> m t
    length :: s -> m Int

batch :: FilePath -> IO [M.Matrix Int]
batch path = do
    bytes <- BS.readFile path
    let chunks = BS.split 3073 bytes
    let vecs = Prelude.map (Prelude.map fromIntegral . BS.unpack . BS.tail) chunks
    let mats = Prelude.map (M.fromList 32 32) vecs
    return mats

label :: FilePath -> IO [Int]
label path = do
    bytes <- BS.readFile path
    let chunks = BS.split 3073 bytes
    let labels = Prelude.map (fromIntegral . BS.head) chunks
    return labels

cifar :: IO CIFAR10
cifar = do
        let path = "data/cifar-10-batches-bin/data_batch_"
        let nums = [1..5]
        let ext = ".bin"
        images <- Prelude.concat <$> mapM (batch . ((path <>) . (++ ext)) . show) nums
        labs <- Prelude.concat <$> mapM (label . ((path <>) . (++ ext)) . show) nums
        return $ CIFAR10 images labs

instance Dataset CIFAR10 IO (M.Matrix Int, Int) where
    (!!) :: CIFAR10 -> Int -> IO (M.Matrix Int, Int)
    (!!) cif i = do
        let image = images cif Prelude.!! i
        let label = labels cif Prelude.!! i
        return (image, label)

    length :: CIFAR10 -> IO Int
    length cif = return $ Prelude.length $ images cif