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

data LayerType t = forall f l. (Functor f, P.Layer l f t) => LayerType l
newtype LayerHandle t = LayerHandle {getLayerType :: IORef (LayerType t) }


consumeModify :: (forall f l. (Functor f, P.Layer l f t) => l -> (l, b)) -> LayerType t -> (LayerType t, b)
consumeModify f (LayerType l) = 
    let (nl, b) = f l in (LayerType nl, b)

class HasLayer a t | a -> t where
    layer :: a -> LayerHandle t

instance HasLayer (LayerHandle t) t where
    layer = id

newLayerHandle :: (Functor f, P.Layer l f t) => l -> IO (LayerHandle t)
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

 
modifyParams :: (HasLayer a t) => a -> (forall f. Functor f => f (Matrix t) -> f(Matrix t)) -> IO ()
modifyParams hand modifier = do
    let ref = getLayerType $ layer hand
    lt <- readIORef ref
    let (newLayer, _) = consumeModify (\l -> (P.setParams l (modifier (P.getParams l)) ,())) lt
    writeIORef ref newLayer




