{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE RankNTypes #-}

module Handle.LayerHandle where

import qualified Data.Layers.Layer as P
import Data.Matrix
import Data.Functor ((<&>))
import Data.IORef(IORef, newIORef, readIORef, writeIORef)

data LayerType t = forall l. (P.Layer l t) => LayerType l
newtype LayerHandle t = LayerHandle {getLayerType :: IORef (LayerType t) }


consume :: (forall l. (P.Layer l t) => l -> b) -> LayerType t -> b
consume f (LayerType l) = f l

consumeModify :: (forall l. (P.Layer l t) => l -> (l, b)) -> LayerType t -> (LayerType t, b)
consumeModify f (LayerType l) = 
    let (nl, b) = f l in (LayerType nl, b)

class HasLayer a t | a -> t where
    layer :: a -> LayerHandle t

instance HasLayer (LayerHandle t) t where
    layer = id

newLayerHandle :: (P.Layer l t) => l -> IO (LayerHandle t)
newLayerHandle x = do
    let lt = LayerType x
    LayerHandle <$> newIORef lt

copyLayer :: LayerHandle t -> IO (LayerHandle t)
copyLayer (LayerHandle ref) = readIORef ref >>= newIORef <&> LayerHandle

forward :: (HasLayer a t, Floating t) => a -> (Matrix t) -> IO (Matrix t)
forward hand inp = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    let (newLayer, res) = consumeModify (flip P.forward inp) lt
    writeIORef ref newLayer
    pure res

backward :: (HasLayer a t, Floating t) => a -> (Matrix t) -> IO (Matrix t)
backward hand inp = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    let (newLayer, res) = consumeModify (flip P.backward inp) lt
    writeIORef ref newLayer
    pure res

 
getParams :: (HasLayer a t) => a -> IO (P.Params (Matrix t))
getParams hand = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    pure $ consume P.getParams lt

setParams :: (HasLayer a t) => a -> P.Params (Matrix t) -> IO()
setParams hand params = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    let (newLayer, _) = consumeModify (\l -> (P.setParams l params, ())) lt
    writeIORef ref newLayer

getGrads :: (HasLayer a t) => a -> IO (P.Params (Matrix t))
getGrads hand = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    pure $ consume P.getGrads lt

setGrads :: (HasLayer a t) => a -> P.Params (Matrix t) -> IO()
setGrads hand params = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    let (newLayer, _) = consumeModify (\l -> (P.setGrads l params, ())) lt
    writeIORef ref newLayer