{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}

module Handle.LayerHandle where

import Data.Layers.Layer
import Data.Matrix
import Data.IORef(IORef, newIORef, readIORef)

data LayerType t = forall f l. (Functor f, Layer l f t) => LayerType l
newtype LayerHandle t = LayerHandle {getLayer :: IORef (LayerType t) }

class HasLayer a t | a -> t where
    layer :: a -> LayerHandle t

instance HasLayer (LayerHandle t) t where
    layer = id

newLayerHandle :: (Functor f, Layer l f t) => l -> IO (LayerHandle t)
newLayerHandle x = do
    let lt = LayerType x
    LayerHandle <$> newIORef lt

forward :: (HasLayer a t) => a -> IO (Matrix t)
forward x = do
    let ref = getLayer $ layer x
    lt <- readIORef ref
    let layer = getLayer lt
    error ""



