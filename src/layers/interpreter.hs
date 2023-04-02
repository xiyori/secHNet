module NN.Interpreter where

import Data.Matrix
import Graph.Graph

class 

forwardOne :: 
  Operation t (Graph (OperationWithForward t) t) -> 
    OperationWithForward t (Graph (OperationWithForward t) t)
backwardOne :: 
  OperationWithForward t (Graph (OperationWithForward t) t) -> 
    OperationWithBackward t (Graph (OperationWithBackward t) t)

forward :: Graph (Operation t) t -> Graph (OperationWithForward t) t