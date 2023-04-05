-- {-# LANGUAGE GADTs #-}
-- {-# LANGUAGE FunctionalDependencies #-}
-- {-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE FlexibleInstances #-}
-- {-# LANGUAGE RankNTypes #-}
-- {-# LANGUAGE AllowAmbiguousTypes #-}

-- module Handle.OptimizerHandle where

-- import qualified Data.Layers.Layer as P
-- import NN.Optimizer(Optimizer, step)
-- import Data.Matrix
-- import Data.Functor ((<&>))
-- import Data.IORef(IORef, newIORef, readIORef, writeIORef)

-- data OptimizerHandle t f = forall o a. Optimizer o t f a => OptimizerHandle {getLayerType :: IORef o }

-- class HasOptimizer o t f| o -> t, o -> f where
--     optimizer :: o -> OptimizerHandle t f

-- instance HasOptimizer (OptimizerHandle t f) t f where
--     optimizer = id

-- newOptimizerHandle :: forall o t f a. (Optimizer o t f a) => o -> IO (OptimizerHandle t f)
-- newOptimizerHandle opt = OptimizerHandle <$> newIORef opt
    
    