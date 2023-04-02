module Optimizer.Optimizer where

import Data.Matrix

-- zeroGrad :: (HasParams t a) => a -> a
-- zeroGrad a = 
--     let zeroMatrix mat = zero (nrows mat) (ncols mat) in
--     setParams $ map $ getParams a
