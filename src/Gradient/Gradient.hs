{-# LANGUAGE ExistentialQuantification #-}
module Gradient.Gradient where
import Data.Matrix
import Layers.Layer

data LayerT t = forall l. (Layer t l) => LayerT l

newtype Network t = Network{ getNetwork :: [LayerT t] }


forwardOne :: ([LayerT t], Matrix t) -> (LayerT t) -> ([LayerT t], Matrix t)
forwardOne (prefix, val) (LayerT l) = 
    let (newLayer, newVal) = forward l val in 
        (prefix ++ [LayerT newLayer], newVal)

forwardNet :: Network t -> Matrix t -> (Network t, Matrix t)
forwardNet (Network layers) input = 
    let (layers, loss) = foldl forwardOne ([], input) layers in (Network layers, loss)

backwardOne :: (LayerT t) -> ([LayerT t], Matrix t) -> ([LayerT t], Matrix t)
backwardOne (LayerT l) (postfix, val) = 
    let (newLayer, newVal) = backward l val in ((LayerT newLayer) : postfix, newVal) 

backwardNet :: Network t -> Matrix t -> (Network t, Matrix t)
backwardNet (Network layers) grad = 
    let (layers, loss) = foldr backwardOne ([], grad) layers in (Network layers, loss)