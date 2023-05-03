{-# LANGUAGE TemplateHaskell #-}

import Control.Monad (unless)
import System.Exit (exitFailure)
import Test.QuickCheck
import Test.Invariant
import qualified Data.Vector.Storable as V
import Instances
import Data.Tensor (
      totalElems,
      equal,
      allClose,
      (&),
      (|.),
      Tensor(Tensor, tensorOffset, tensorData, shape),
      HasArange(arange),
      Indexer(I, S, E, (:.), Ell, None, A),
      Shape,
      fromList3,
      tensor,
      full,
      zeros,
      ones,
      scalar,
      single,
      zerosLike,
      onesLike,
      eye,
      broadcast,
      (//),
      (%),
      elementwise,
      (!),
      (!:),
      item,
      numel,
      mean,
      copy,
      astype,
      transpose,
      flatten,
      view,
      insertDim)
import qualified Data.Tensor as T

import Foreign.C.Types
import System.IO.Unsafe


-- Functional
-- ----------

prop_fromList3 :: Bool
prop_fromList3 = (fromList3 [[[], []], [[], []], [[], []]] :: Tensor CFloat) `equal` zeros (V.fromList [3, 2, 0])

prop_tensor0 :: Shape -> Bool
prop_tensor0 shape = tensor shape (const (0 :: CFloat)) `equal` zeros shape

prop_tensor :: Shape -> Gen Bool
prop_tensor shape = do
  x@(Tensor _ _ _ dat) <- arbitraryContiguousWithShape shape :: Gen (Tensor CFloat)
  return $ tensor shape (dat V.!) `equal` x

prop_empty :: Bool
prop_empty = (T.empty :: Tensor CFloat) `equal` full (V.singleton 0) undefined

prop_scalar :: CFloat -> Bool
prop_scalar value = item (scalar value) == value

prop_single :: CFloat -> Bool
prop_single value = item (single value) == value

prop_eye_unit :: Bool
prop_eye_unit = x ! [0, 0] == 1 && x ! [1, 1] == 1 && x ! [0, 1] == 0 && x ! [1, 0] == 0
  where
    x = eye 2 2 0 :: Tensor CFloat

prop_eye_transpose0_unit :: Bool
prop_eye_transpose0_unit = transpose x `equal` x
  where
    x = eye 3 3 0 :: Tensor CFloat

prop_eye_transpose1_unit :: Bool
prop_eye_transpose1_unit = not $ transpose x `equal` x
  where
    x = eye 3 3 1 :: Tensor CFloat

prop_arange :: Shape -> Bool
prop_arange shape
  | totalElems shape /= 0 =
    flatten (tensor shape fromIntegral) `equal`
    arange (0 :: CFloat) (fromIntegral $ totalElems shape) 1
  | otherwise =
    True

prop_arange_neg :: Shape -> Bool
prop_arange_neg shape
  | totalElems shape /= 0 =
    flatten (tensor shape $ negate . fromIntegral) `allClose`
    arange (0 :: CFloat) (-(fromIntegral $ totalElems shape)) (-1)
  | otherwise =
    True

prop_arange_single :: Bool
prop_arange_single = (arange 0 1 2 :: Tensor CFloat) `equal` single 0

prop_arange_single_neg :: Bool
prop_arange_single_neg = (arange 0 (-1) (-2) :: Tensor CFloat) `equal` single 0

prop_arange_empty :: Bool
prop_arange_empty = (arange 0 (-1) 1 :: Tensor CFloat) `equal` T.empty

prop_arange_empty_neg :: Bool
prop_arange_empty_neg = (arange 0 1 (-1) :: Tensor CFloat) `equal` T.empty

prop_arange_int :: Shape -> Bool
prop_arange_int shape
  | totalElems shape /= 0 =
    flatten (tensor shape fromIntegral) `equal`
    arange (0 :: CLLong) (fromIntegral $ totalElems shape) 1
  | otherwise =
    True

prop_arange_neg_int :: Shape -> Bool
prop_arange_neg_int shape
  | totalElems shape /= 0 =
    flatten (tensor shape $ negate . fromIntegral) `equal`
    arange (0 :: CLLong) (-(fromIntegral $ totalElems shape)) (-1)
  | otherwise =
    True

prop_arange_single_int :: Bool
prop_arange_single_int = (arange 0 1 2 :: Tensor CLLong) `equal` single 0

prop_arange_single_neg_int :: Bool
prop_arange_single_neg_int = (arange 0 (-1) (-2) :: Tensor CLLong) `equal` single 0

prop_arange_empty_int :: Bool
prop_arange_empty_int = (arange 0 (-1) 1 :: Tensor CLLong) `equal` T.empty

prop_arange_empty_neg_int :: Bool
prop_arange_empty_neg_int = (arange 0 1 (-1) :: Tensor CLLong) `equal` T.empty

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

prop_elementwise_id :: Shape -> Bool
prop_elementwise_id shape = elementwise (*) (ones shape :: Tensor CFloat) (ones shape) `equal` ones shape

prop_elementwise_commutative :: Gen Bool
prop_elementwise_commutative = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ elementwise (+) x1 x2 `allClose` elementwise (+) x2 x1

prop_int_div :: Gen Bool  -- Fails with arithmetic overflow for *_MIN or *_MAX ?
prop_int_div = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CChar, Tensor CChar)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ x1 // x2 `equal` elementwise div x1 x2
  }

prop_int_div_float :: Gen Bool
prop_int_div_float = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ x1 // x2 `equal` elementwise (\ a b -> fromIntegral $ floor $ a / b) x1 x2
  }

prop_mod :: Gen Bool
prop_mod = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CChar, Tensor CChar)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ x1 % x2 `equal` elementwise mod x1 x2
  }

prop_mod_float :: Gen Bool
prop_mod_float = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ x1 % x2 `equal` elementwise (\ a b -> a - b * fromIntegral (floor $ a / b)) x1 x2
  }

prop_getelem :: Shape -> Bool
prop_getelem shape
  | totalElems shape /= 0 = zeros shape ! replicate (V.length shape) 0 == (0 :: CFloat)
  | otherwise             = True

prop_getelem_scalar ::  Bool
prop_getelem_scalar = scalar (0 :: CFloat) ! [] == 0

prop_getelem_single ::  Bool
prop_getelem_single = single (0 :: CFloat) ! [0] == 0

prop_slice_index :: Tensor CFloat -> Bool
prop_slice_index x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = scalar (x ! replicate (V.length shape) 0) `equal` x !: replicate (V.length shape) 0
  | otherwise             = True

prop_slice_single :: Tensor CFloat -> Bool
prop_slice_single x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = x ! replicate (V.length shape) 0 == item (x !: replicate (V.length shape) (0:.1))
  | otherwise             = True

prop_slice_all :: Tensor CFloat -> Bool
prop_slice_all x@(Tensor shape _ _ _) = x !: replicate (V.length shape) A `equal` x

prop_slice_all0 :: Tensor CFloat -> Bool
prop_slice_all0 x@(Tensor shape _ _ _) = x !: [] `equal` x

prop_slice_all1 :: Tensor CFloat -> Bool
prop_slice_all1 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A] `equal` x
  | otherwise          = True

prop_slice_start_all :: Tensor CFloat -> Bool
prop_slice_start_all x@(Tensor shape _ _ _) = x !: replicate (V.length shape) (S 0) `equal` x

prop_slice_start1_all :: Tensor CFloat -> Bool
prop_slice_start1_all x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [S 0] `equal` x
  | otherwise          = True

prop_slice_start_equiv :: Tensor CFloat -> Bool
prop_slice_start_equiv x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [S 1] `equal` x !: [1:.fromIntegral (V.head shape)]
  | otherwise          = True

prop_slice_end_all :: Tensor CFloat -> Bool
prop_slice_end_all x@(Tensor shape _ _ _) = x !: map (E . fromIntegral) (V.toList shape) `equal` x

prop_slice_end1_all :: Tensor CFloat -> Bool
prop_slice_end1_all x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [E $ fromIntegral (V.head shape)] `equal` x
  | otherwise          = True

prop_slice_end_equiv :: Tensor CFloat -> Bool
prop_slice_end_equiv x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [E (-1)] `equal` x !: [0 :. fromIntegral (V.head shape) - 1]
  | otherwise          = True

prop_slice_negative :: Tensor CFloat -> Bool
prop_slice_negative x@(Tensor shape _ _ _)
  | V.length shape > 0 =
    x !: [-3:. -1] `equal` x !: [max 0 (fromIntegral (V.head shape) - 3) :. fromIntegral (V.head shape) - 1]
  | otherwise          = True

prop_slice_none :: Tensor CFloat -> Bool
prop_slice_none x = insertDim x 0 `equal` x !: [None]

prop_slice_ell :: Tensor CFloat -> Bool
prop_slice_ell x@(Tensor shape _ _ _)
  | V.length shape > 0 =
    x !: [Ell, 0:.1] `equal`
    x !: (replicate (V.length shape - 1) A ++ [0 :. 1])
  | otherwise =
    True

prop_slice_insert_dim :: Tensor CFloat -> Bool
prop_slice_insert_dim x = insertDim x (-1) `equal` x !: [Ell, None]

prop_slice_step1 :: Tensor CFloat -> Bool
prop_slice_step1 x@(Tensor shape _ _ _) = x !: replicate (V.length shape) (A :. 1) `equal` x

prop_slice_step11 :: Tensor CFloat -> Bool
prop_slice_step11 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A :. 1] `equal` x
  | otherwise          = True

prop_slice_neg_step_id :: Tensor CFloat -> Bool
prop_slice_neg_step_id x@(Tensor shape _ _ _) = x !: slice !: slice `equal` x
  where
    slice = replicate (V.length shape) (A :. -1)

prop_slice_neg_step_id1 :: Tensor CFloat -> Bool
prop_slice_neg_step_id1 x@(Tensor shape _ _ _)
  | V.length shape > 0 = x !: [A :. -1] !: [A :. -1] `equal` x
  | otherwise          = True

prop_slice_step :: Bool
prop_slice_step = (arange 0 10 1 :: Tensor CFloat) !: [A :. 2] `equal` arange 0 10 2

prop_slice_neg_step :: Bool
prop_slice_neg_step = (arange 0 10 1 :: Tensor CFloat) !: [A :. -2] `allClose` arange 9 (-1) (-2)

prop_numel :: Shape -> Bool
prop_numel shape = fromIntegral (numel x) == V.length (tensorData x)
  where
    x = copy $ zeros shape :: Tensor CFloat

prop_min :: Tensor CDouble -> Bool
prop_min x
  | numel x /= 0 =
    T.min x == T.foldl' min (1 / 0) x
  | otherwise =
    True

prop_max :: Tensor CDouble -> Bool
prop_max x
  | numel x /= 0 =
    T.max x == T.foldl' max (-1 / 0) x
  | otherwise =
    True

prop_sum :: Shape -> Bool
prop_sum shape = T.sum (ones shape :: Tensor CFloat) == fromIntegral (totalElems shape)

prop_sum_int :: Tensor CFloat -> Bool
prop_sum_int x = T.sum xI == T.foldl' (+) 0 xI
  where
    xI = astype x :: Tensor CLLong

prop_sum_distributive :: Shape -> Gen Bool
prop_sum_distributive shape = do
  (x1, x2) <- arbitraryPairWithShape shape :: Gen (Tensor CDouble, Tensor CDouble)
  return $ scalar (T.sum x1 + T.sum x2) `allClose` scalar (T.sum (x1 + x2))

prop_sum_empty :: Shape -> Bool
prop_sum_empty shape = T.sum (ones $ V.snoc shape 0) == (0 :: CFloat)

prop_sum_scalar :: Bool
prop_sum_scalar = T.sum (scalar 1) == (1 :: CFloat)

prop_sum_single :: Bool
prop_sum_single = T.sum (single 1) == (1 :: CFloat)

prop_mean_empty :: Shape -> Bool
prop_mean_empty shape = isNaN $ T.mean (ones $ V.snoc shape 0 :: Tensor CFloat)

prop_copy :: Tensor CFloat -> Bool
prop_copy x = x `equal` copy x

prop_copy_offset :: Tensor CFloat -> Bool
prop_copy_offset x = tensorOffset (copy x) == 0

prop_copy_numel :: Tensor CFloat -> Bool
prop_copy_numel x = fromIntegral (numel x1) == V.length (tensorData x1)
  where
    x1 = copy x

prop_astype :: Tensor CFloat -> Bool
prop_astype x = (astype x :: Tensor CLLong) `equal` T.map (
    \ value ->
      if value < 0 then
        ceiling value
      else floor value
  ) x

prop_astype_uint :: Tensor CFloat -> Bool
prop_astype_uint x = (astype xLL :: Tensor CULLong) `equal` T.map fromIntegral xLL
  where
    xLL = astype x :: Tensor CLLong

prop_transpose_id :: Tensor CFloat -> Bool
prop_transpose_id x = transpose (transpose x1) `equal` x1
  where
    x1 = insertDim (insertDim x 0) 0

prop_flatten_id :: Tensor CFloat -> Bool
prop_flatten_id x = flatten x `view` shape x `equal` x

prop_view :: Tensor CFloat -> Bool
prop_view x =
  transpose (
    (xT `view` V.singleton (totalElems $ shape x)) `view` shape xT
  ) `equal` x1
  where
    x1 = insertDim (insertDim x 0) 0
    xT = transpose x1

prop_insert_dim :: Shape -> Bool
prop_insert_dim shape = insertDim x 0 `equal` zeros (V.concat [V.singleton 1, shape])
  where
    x = zeros shape :: Tensor CFloat


-- Boolean
-- -------

prop_equal :: Tensor CFloat -> Bool
prop_equal x = x `equal` x

prop_not_equal :: Tensor CFloat -> Bool
prop_not_equal x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = not $ x `equal` (x + 1)
  | otherwise             = True

prop_allclose :: Tensor CFloat -> Bool
prop_allclose x = allClose x x

prop_not_allclose :: Tensor CFloat -> Bool
prop_not_allclose x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = not $ allClose x (x + 1)
  | otherwise             = True

prop_tensor_equal1 :: Tensor CFloat -> Bool
prop_tensor_equal1 x = (x T.== x) `equal` onesLike x

prop_tensor_equal0 :: Tensor CFloat -> Bool
prop_tensor_equal0 x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = (x T.== x + 1) `equal` zerosLike x
  | otherwise             = True

prop_tensor_not_equal0 :: Tensor CFloat -> Bool
prop_tensor_not_equal0 x = (x T./= x) `equal` zerosLike x

prop_tensor_not_equal1 :: Tensor CFloat -> Bool
prop_tensor_not_equal1 x@(Tensor shape _ _ _)
  | totalElems shape /= 0 = (x T./= x + 1) `equal` onesLike x
  | otherwise             = True

prop_tensor_less :: Tensor CFloat -> Bool
prop_tensor_less x = (x T.< x) `equal` zerosLike x

prop_tensor_leq :: Tensor CFloat -> Bool
prop_tensor_leq x = (x T.<= x) `equal` onesLike x

prop_tensor_not_greater :: Gen Bool
prop_tensor_not_greater = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ T.not (x1b T.> x2b) `equal` (x1b T.<= x2b)
  }

prop_not :: Shape -> Bool
prop_not shape = T.not (zeros shape) `equal` (ones shape :: Tensor CBool)

prop_and :: Shape -> Bool
prop_and shape = (ones shape & ones shape) `equal` (ones shape :: Tensor CBool) &&
                 (zeros shape & ones shape) `equal` (zeros shape :: Tensor CBool) &&
                 (zeros shape & zeros shape) `equal` (zeros shape :: Tensor CBool)

prop_or :: Shape -> Bool
prop_or shape = (ones shape |. ones shape) `equal` (ones shape :: Tensor CBool) &&
                (zeros shape |. ones shape) `equal` (ones shape :: Tensor CBool) &&
                (zeros shape |. zeros shape) `equal` (zeros shape :: Tensor CBool)


-- NumTensor
-- ---------

prop_abs :: Shape -> Bool
prop_abs shape = abs x `equal` x
  where
    x = ones shape :: Tensor CFloat

prop_abs_neg :: Shape -> Bool
prop_abs_neg shape = abs x `equal` (-x)
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

prop_num_associative :: Shape -> Gen Bool
prop_num_associative shape = do
  (x1, x2) <- arbitraryPairWithShape shape :: Gen (Tensor CDouble, Tensor CDouble)
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

prop_show_eye_shift :: Bool
prop_show_eye_shift =
  show (eye 4 3 (-1) :: Tensor CFloat) ==
  "tensor([[0., 0., 0.],\n" ++
  "        [1., 0., 0.],\n" ++
  "        [0., 1., 0.],\n" ++
  "        [0., 0., 1.]])"

prop_show_range :: Bool
prop_show_range =
  show (arange 0 1001 1 :: Tensor CFloat) ==
  "tensor([   0.,    1.,    2., ...,  998.,  999., 1000.])"

prop_show_empty :: Bool
prop_show_empty =
  show (zeros (V.fromList [1, 2, 0]) :: Tensor CFloat) ==
  "tensor([], shape=[1,2,0], dtype=float32)"

return []

main :: IO ()
main = do
  success <- $quickCheckAll
  -- success <- $forAllProperties $ quickCheckWithResult (stdArgs {maxSuccess = 10000})
  unless success exitFailure
