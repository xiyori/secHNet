{-# LANGUAGE TemplateHaskell #-}

import Control.Monad (unless)
import System.Exit (exitFailure)
import Test.QuickCheck
import Test.Invariant
import qualified Data.Vector.Storable as V
import Instances
import Data.Tensor.Index
import Data.Tensor as T

import Foreign.C.Types


-- Functional
-- ----------

prop_tensor0 :: Index -> Bool
prop_tensor0 shape = tensor shape (const (0 :: CFloat)) == zeros shape

prop_tensor :: Tensor CFloat -> Bool
prop_tensor x@(Tensor shape _ _ dat) = tensor shape (dat V.!) == x

prop_scalar :: CFloat -> Bool
prop_scalar value = item (scalar value) == value

prop_single :: CFloat -> Bool
prop_single value = item (single value) == value

prop_broadcast :: Tensor CFloat -> Bool
prop_broadcast x = shape (snd $ broadcast x (scalar (0 :: CFloat))) == shape x

prop_elementwise_zero :: Tensor CFloat -> Bool
prop_elementwise_zero x = elementwise (*) x 0 == zerosLike x

prop_elementwise_id :: Index -> Bool
prop_elementwise_id shape = elementwise (*) (ones shape :: Tensor CFloat) (ones shape) == ones shape

prop_elementwise_commutative :: Tensor CFloat -> Tensor CFloat -> Bool
prop_elementwise_commutative = commutative (elementwise (+))

prop_eq :: Tensor CFloat -> Bool
prop_eq x = x == x

prop_not_eq :: Tensor CFloat -> Tensor CFloat -> Bool
prop_not_eq x y = x /= y

prop_getelem :: Index -> Bool
prop_getelem shape
  | totalElems shape /= 0 = zeros shape ! replicate (V.length shape) 0 == (0 :: CFloat)
  | otherwise             = True

prop_getelem_scalar ::  Bool
prop_getelem_scalar = scalar (0 :: CFloat) ! [] == 0

prop_getelem_single ::  Bool
prop_getelem_single = single (0 :: CFloat) ! [0] == 0

prop_numel :: Index -> Bool
prop_numel shape = fromIntegral (numel x) == V.length (tensorData x)
  where
    x = zeros shape :: Tensor CFloat

prop_copy :: Tensor CFloat -> Bool
prop_copy x = x == copy x

prop_copy_offset :: Tensor CFloat -> Bool
prop_copy_offset x = offset (copy x) == 0

prop_copy_numel :: Tensor CFloat -> Bool
prop_copy_numel x = fromIntegral (numel x1) == V.length (tensorData x1)
  where
    x1 = copy x

prop_transpose_id :: Tensor CFloat -> Bool
prop_transpose_id x = transpose (transpose x) == x

-- prop_flatten - need reshape

prop_insert_dim :: Index -> Bool
prop_insert_dim shape = insertDim x 0 == zeros (V.concat [V.singleton 1, shape])
  where
    x = zeros shape :: Tensor CFloat


-- FloatTensor
-- -----------

prop_eye_unit :: Bool
prop_eye_unit = x ! [0, 0] == 1 && x ! [1, 1] == 1 && x ! [0, 1] == 0 && x ! [1, 0] == 0
  where
    x = eye 2 2 0 :: Tensor CFloat

prop_eye_transpose0_unit :: Bool
prop_eye_transpose0_unit = transpose x == x
  where
    x = eye 3 3 0 :: Tensor CFloat

prop_eye_transpose1_unit :: Bool
prop_eye_transpose1_unit = transpose x /= x
  where
    x = eye 3 3 1 :: Tensor CFloat

prop_arange :: Index -> Bool
prop_arange shape =
  flatten (tensor shape fromIntegral) ==
  arange (0 :: CFloat) (fromIntegral $ totalElems shape) 1

prop_sum :: Index -> Bool
prop_sum shape = T.sum (ones shape :: Tensor CFloat) == fromIntegral (totalElems shape)

prop_sum_associative :: Tensor CFloat -> Tensor CFloat -> Bool
prop_sum_associative x1 x2 = T.sum x1 + T.sum x2 == T.sum (x1 + x2)

prop_abs :: Index -> Bool
prop_abs shape = abs x == x
  where
    x = ones shape :: Tensor CFloat

prop_abs_neg :: Index -> Bool
prop_abs_neg shape = abs x == (-x)
  where
    x = full shape (-1) :: Tensor CFloat
  
prop_add_commutative :: Tensor CFloat -> Tensor CFloat -> Bool
prop_add_commutative = commutative (+)

prop_num_associative :: Tensor CFloat -> Tensor CFloat -> Tensor CFloat -> Bool
prop_num_associative = (*) `distributesLeftOver` (+)

return []

main :: IO ()
main = do
  success <- $quickCheckAll
  unless success exitFailure
