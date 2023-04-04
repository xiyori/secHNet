{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ConstrainedClassMethods #-}

module NN.Optimizer where

import Data.Matrix
import Data.Maybe (fromJust)

class Applicative f => Optimizer o f t a where
    step :: o -> f (Matrix t) -> f (Matrix t) -> f a -> (o, f (Matrix t), f a)

instance Optimizer Momentum 

                    dW = "dW" ++ show (l + 1)
                    db = "db" ++ show (l + 1)
                    v_dW = fromJust $ lookup dW v
                    v_db = fromJust $ lookup db v
                    dW_grads = fromJust $ lookup (dW ++ "_grads") grads
                    db_grads = fromJust $ lookup (db ++ "_grads") grads
                    new_v_dW = scaleMatrix beta v_dW + scaleMatrix (1 - beta) dW_grads
                    new_v_db = scaleMatrix beta v_db + scaleMatrix (1 - beta) db_grads

                    w = "W" ++ show (l + 1)
                    b = "b" ++ show (l + 1)
                    w_params = fromJust $ lookup w params
                    b_params = fromJust $ lookup b params
                    new_W_params = w_params - scaleMatrix learningRate new_v_dW
                    new_b_params = b_params - scaleMatrix learningRate new_v_db

