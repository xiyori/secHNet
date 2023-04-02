module Layers.Interpreter where

import Control.Monad.Free
import Data.Matrix
import Graph.Graph

data ExecutionResult t = ExecutionResult {getResult :: t , getGraph :: (Graph (Operation t) t)}

instance Functor (ExecutionResult t) where
    

forward :: Graph (Operation t) t -> ExecutionResult t
forward graph = foldA foldOne graph where
    foldOne :: Operation t (t, Graph (Operation t) t) -> (t, Graph (Operation t) t)