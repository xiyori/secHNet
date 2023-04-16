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
import System.IO.Unsafe


-- Functional
-- ----------

prop_tensor0 :: Index -> Bool
prop_tensor0 shape = tensor shape (const (0 :: CFloat)) == zeros shape

prop_tensor :: Index -> Gen Bool
prop_tensor shape = do
  x@(Tensor _ _ _ dat) <- arbitraryContiguousWithShape shape :: Gen (Tensor CFloat)
  return $ tensor shape (dat V.!) == x

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

prop_not_eq :: Tensor CFloat -> Bool
prop_not_eq x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = x /= (x + 1)
  | otherwise             = True

prop_allclose :: Tensor CFloat -> Bool
prop_allclose x = allClose x x

prop_not_allclose :: Tensor CFloat -> Bool
prop_not_allclose x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = not $ allClose x (x + 1)
  | otherwise             = True

prop_getelem :: Index -> Bool
prop_getelem shape
  | totalElems shape /= 0 = zeros shape ! replicate (V.length shape) 0 == (0 :: CFloat)
  | otherwise             = True

prop_getelem_scalar ::  Bool
prop_getelem_scalar = scalar (0 :: CFloat) ! [] == 0

prop_getelem_single ::  Bool
prop_getelem_single = single (0 :: CFloat) ! [0] == 0

prop_slice_index :: Tensor CFloat -> Bool
prop_slice_index x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = x ! replicate (V.length shape) 0 == item (x !: replicate (V.length shape) (I 0))
  | otherwise             = True

prop_slice_single :: Tensor CFloat -> Bool
prop_slice_single x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = x ! replicate (V.length shape) 0 == item (x !: replicate (V.length shape) (0:.1))
  | otherwise             = True

prop_slice_all :: Tensor CFloat -> Bool
prop_slice_all x@(Tensor shape _ _ _) = x !: replicate (V.length shape) A == x

prop_slice_all0 :: Tensor CFloat -> Bool
prop_slice_all0 x@(Tensor shape _ _ _) = x !: [] == x

prop_slice_all1 :: Tensor CFloat -> Bool
prop_slice_all1 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A] == x
  | otherwise          = True

prop_slice_start_all :: Tensor CFloat -> Bool
prop_slice_start_all x@(Tensor shape _ _ _) = x !: replicate (V.length shape) (S 0) == x

prop_slice_start1_all :: Tensor CFloat -> Bool
prop_slice_start1_all x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [S 0] == x
  | otherwise          = True

prop_slice_start_equiv :: Tensor CFloat -> Bool
prop_slice_start_equiv x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [S 1] == x !: [1:.V.head shape]
  | otherwise          = True

prop_slice_end_all :: Tensor CFloat -> Bool
prop_slice_end_all x@(Tensor shape _ _ _) = x !: Prelude.map E (V.toList shape) == x

prop_slice_end1_all :: Tensor CFloat -> Bool
prop_slice_end1_all x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [E $ V.head shape] == x
  | otherwise          = True

prop_slice_end_equiv :: Tensor CFloat -> Bool
prop_slice_end_equiv x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [E (-1)] == x !: [0:.V.head shape - 1]
  | otherwise          = True

prop_slice_negative :: Tensor CFloat -> Bool
prop_slice_negative x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [-3:. -1] == x !: [max 0 (V.head shape - 3):.V.head shape - 1]
  | otherwise          = True

prop_slice_none :: Tensor CFloat -> Bool
prop_slice_none x = insertDim x 0 == x !: [None]

prop_slice_ell :: Tensor CFloat -> Bool
prop_slice_ell x@(Tensor shape _ _ _)
  | V.length shape > 0 =
    x !: [Ell, 0:.1] ==
    x !: (replicate (V.length shape - 1) A ++ [0:.1])
  | otherwise =
    True

prop_slice_insert_dim :: Tensor CFloat -> Bool
prop_slice_insert_dim x = insertDim x (-1) == x !: [Ell, None]

prop_slice_step1 :: Tensor CFloat -> Bool
prop_slice_step1 x@(Tensor shape _ _ _) = x !: replicate (V.length shape) (A:|1) == x

prop_slice_step11 :: Tensor CFloat -> Bool
prop_slice_step11 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A:|1] == x
  | otherwise          = True

prop_slice_neg_step_id :: Tensor CFloat -> Bool
prop_slice_neg_step_id x@(Tensor shape _ _ _) = x !: slice !: slice == x
  where
    slice = replicate (V.length shape) (A:| -1)

prop_slice_neg_step_id1 :: Tensor CFloat -> Bool
prop_slice_neg_step_id1 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A:| -1] !: [A:| -1] == x
  | otherwise          = True

prop_slice_step :: Bool
prop_slice_step = (arange 0 10 1 :: Tensor CFloat) !: [A:|2] == arange 0 10 2

prop_slice_neg_step :: Bool
prop_slice_neg_step = (arange 0 10 1 :: Tensor CFloat) !: [A:| -2] `allClose` arange 9 (-1) (-2)

prop_numel :: Index -> Bool
prop_numel shape = fromIntegral (numel x) == V.length (tensorData x)
  where
    x = zeros shape :: Tensor CFloat

prop_copy :: Tensor CFloat -> Bool
prop_copy x = x == copy x

prop_copy_offset :: Tensor CFloat -> Bool
prop_copy_offset x = tensorOffset (copy x) == 0

prop_copy_numel :: Tensor CFloat -> Bool
prop_copy_numel x = fromIntegral (numel x1) == V.length (tensorData x1)
  where
    x1 = copy x

prop_transpose_id :: Tensor CFloat -> Bool
prop_transpose_id x = transpose (transpose x1) == x1
  where
    x1 = insertDim (insertDim x 0) 0

prop_flatten_id :: Tensor CFloat -> Bool
prop_flatten_id x = flatten x `view` shape x == x

prop_view :: Tensor CFloat -> Bool
prop_view x =
  transpose (
    (xT `view` V.singleton (totalElems $ shape x)) `view` shape xT
  ) == x1
  where
    x1 = insertDim (insertDim x 0) 0
    xT = transpose x1

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

prop_arange_neg :: Index -> Bool
prop_arange_neg shape
  | totalElems shape /= 0 =
    flatten (tensor shape fromIntegral) `allClose`
    (-arange (0 :: CFloat) (fromIntegral $ -totalElems shape) (-1))
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

prop_num_associative :: Index -> Gen Bool
prop_num_associative shape = do
  (x1, x2) <- arbitraryPairWithShape shape :: Gen (Tensor CFloat, Tensor CFloat)
  return $ ((x1 + x2) * x2) `allClose` (x1 * x2 + x2 * x2)


-- Instances
-- ---------

prop_show_scalar :: Bool
prop_show_scalar = show (scalar (0 :: CFloat)) == "tensor(0.)"

prop_show_single :: Bool
prop_show_single = show (single (0 :: CFloat)) == "tensor([0.])"

prop_show_eye :: Bool
prop_show_eye =
  show (eye 3 3 0 :: Tensor CFloat) ==
  "tensor([[1., 0., 0.],\n" ++
  "        [0., 1., 0.],\n" ++
  "        [0., 0., 1.]])"

prop_show_range :: Bool
prop_show_range =
  show (arange 0 1001 1 :: Tensor CFloat) ==
  "tensor([   0.,    1.,    2., ...,  998.,  999., 1000.])"

prop_show_empty :: Bool
prop_show_empty =
  show (zeros (V.fromList [1, 2, 0]) :: Tensor CFloat) ==
  "tensor([], shape=[1,2,0], dtype=float)"

return []

main :: IO ()
main = do
  success <- $quickCheckAll
  -- success <- $forAllProperties $ quickCheckWithResult (stdArgs {maxSuccess = 10000})
  unless success exitFailure
