{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE FlexibleInstances #-}

module NN.Optimizer where

import Control.Applicative (liftA3)
import Data.Traversable (sequenceA)
import Data.Tensor
import Data.Maybe (fromJust)
import Foreign.C.Types

class (Applicative f) => Optimizer o f t a where
    step :: o -> f (Tensor t) -> f (Tensor t) -> f a -> (o, f (Tensor t), f a)

data Momentum = Momentum {beta :: CFloat, learningRate :: CFloat}

instance (Applicative f) => Optimizer Momentum f CFloat (Tensor CFloat) where
    step optim params grads v = 
        let tup_func = liftA3 helper params grads v in (optim, fmap fst tup_func, fmap snd tup_func)
            where
                helper :: Tensor CFloat -> Tensor CFloat -> Tensor CFloat -> (Tensor CFloat, Tensor CFloat)
                helper param grad v' = 
                    let new_vels = (scalar $ beta optim) * v' + (scalar $ 1 - (beta optim)) * grad
                        new_params = param - (scalar $ learningRate optim) * new_vels
                    in (new_params, new_vels)
