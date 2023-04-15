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

prop_broadcast_scalar :: Tensor CFloat -> Bool
prop_broadcast_scalar x = shape (snd $ broadcast x (scalar (0 :: CFloat))) == shape x

prop_broadcast :: Gen Bool
prop_broadcast = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ shape x1b == shape x2b
  }

prop_elementwise_zero :: Tensor CFloat -> Bool
prop_elementwise_zero x = elementwise (*) x 0 `allClose` zerosLike x

prop_elementwise_id :: Index -> Bool
prop_elementwise_id shape = elementwise (*) (ones shape :: Tensor CFloat) (ones shape) == ones shape

prop_elementwise_commutative :: Gen Bool
prop_elementwise_commutative = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ elementwise (+) x1 x2 `allClose` elementwise (+) x2 x1

prop_eq :: Tensor CFloat -> Bool
prop_eq x = x == x

prop_not_eq :: Tensor CFloat -> Tensor CFloat -> Bool
prop_not_eq x1 x2 = x1 /= x2

prop_allclose :: Tensor CFloat -> Bool
prop_allclose x = allClose x x

prop_not_allclose :: Tensor CFloat -> Tensor CFloat -> Bool
prop_not_allclose x1 x2 = not $ allClose x1 x2

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
prop_transpose_id x = transpose (transpose x1) == x1
  where
    x1 = insertDim (insertDim x 0) 0

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
prop_arange shape
  | totalElems shape /= 0 =
    flatten (tensor shape fromIntegral) ==
    arange (0 :: CFloat) (fromIntegral $ totalElems shape) 1
  | otherwise =
    True

prop_sum :: Index -> Bool
prop_sum shape = T.sum (ones shape :: Tensor CFloat) == fromIntegral (totalElems shape)

prop_sum_distributive :: Index -> Gen Bool
prop_sum_distributive shape = do
  (x1, x2) <- arbitraryPairWithShape shape :: Gen (Tensor CFloat, Tensor CFloat)
  return $ scalar (T.sum x1 + T.sum x2) `allClose` scalar (T.sum (x1 + x2))

prop_abs :: Index -> Bool
prop_abs shape = abs x == x
  where
    x = ones shape :: Tensor CFloat

prop_abs_neg :: Index -> Bool
prop_abs_neg shape = abs x == (-x)
  where
    x = full shape (-1) :: Tensor CFloat

prop_exp :: Tensor CFloat -> Bool
prop_exp x = exp x `allClose` T.map exp x

prop_sin :: Tensor CFloat -> Bool
prop_sin x = sin x `allClose` T.map sin x

prop_add_commutative :: Gen Bool
prop_add_commutative = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ (x1 + x2) `allClose` (x2 + x1)

prop_num_associative :: Gen Bool
prop_num_associative = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ ((x1 + x2) * x2) `allClose` (x1 * x2 + x2 * x2)

return []

main :: IO ()
main = do
  success <- $quickCheckAll
  unless success exitFailure
