{-# LANGUAGE GADTs #-}
module Gradient.Gradient where

import Layers.Layer

newtype LayerT t where
    LayerT :: (Layer l t) => l -> LayerT t

newtype Network t = { getNetwork :: [LayerT t] }

forward :: Network t -> t -> (Network t, t)
forward (Network layers) input = 
    let res = foldr  [[], input]