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
      Tensor(Tensor, tensorOffset, tensorData),
      HasArange(arange),
      Indexer(I, A, S, E, (:.), T, Ell, None),
      Index,
      fromList,
      fromList2,
      fromList3,
      fromList4,
      fromList5,
      tensor,
      tensor_,
      full,
      zeros,
      ones,
      scalar,
      single,
      zerosLike,
      onesLike,
      eye,
      broadcast,
      broadcastN,
      (//),
      (%),
      (@),
      elementwise,
      (!),
      (!:),
      item,
      dim,
      shape,
      numel,
      mean,
      sumAlongDims,
      sumAlongDim,
      copy,
      astype,
      transpose,
      flatten,
      view,
      insertDim,
      insertDims,
      swapDims)
import qualified Data.Tensor as T

import Foreign.C.Types
import System.IO.Unsafe


-- Functional
-- ----------

prop_fromList3 :: Bool
prop_fromList3 = (fromList3 [[[], []], [[], []], [[], []]] :: Tensor CFloat) `equal` zeros [3, 2, 0]

prop_tensor0 :: Index -> Bool
prop_tensor0 shape = tensor shape (const (0 :: CFloat)) `equal` zeros shape

prop_tensor :: Tensor CFloat -> Bool
prop_tensor x = tensor (shape x) (x !) `equal` x

prop_empty :: Bool
prop_empty = (T.empty :: Tensor CFloat) `equal` full [0] undefined

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

prop_arange :: Index -> Bool
prop_arange shape
  | totalElems shape /= 0 =
    flatten (tensor_ (V.map fromIntegral $ V.fromList shape) fromIntegral) `equal`
    arange (0 :: CFloat) (fromIntegral $ totalElems shape) 1
  | otherwise =
    True

prop_arange_neg :: Index -> Bool
prop_arange_neg shape
  | totalElems shape /= 0 =
    flatten (tensor_ (V.map fromIntegral $ V.fromList shape) $ negate . fromIntegral) `allClose`
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

prop_arange_int :: Index -> Bool
prop_arange_int shape
  | totalElems shape /= 0 =
    flatten (tensor_ (V.map fromIntegral $ V.fromList shape) fromIntegral) `equal`
    arange (0 :: CLLong) (fromIntegral $ totalElems shape) 1
  | otherwise =
    True

prop_arange_neg_int :: Index -> Bool
prop_arange_neg_int shape
  | totalElems shape /= 0 =
    flatten (tensor_ (V.map fromIntegral $ V.fromList shape) $ negate . fromIntegral) `equal`
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
prop_broadcast_scalar x = shape (last $ broadcastN [x, scalar 0]) == shape x

prop_broadcast :: Gen Bool
prop_broadcast = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  case broadcast x1 x2 of {(x1b, x2b) ->
    return $ shape x1b == shape x2b
  }

prop_elementwise_zero :: Tensor CFloat -> Bool
prop_elementwise_zero x = elementwise (*) x 0 `allClose` zerosLike x

prop_elementwise_id :: Index -> Bool
prop_elementwise_id shape = elementwise (*) (ones shape :: Tensor CFloat) (ones shape) `equal` ones shape

prop_elementwise_commutative :: Gen Bool
prop_elementwise_commutative = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ elementwise (+) x1 x2 `allClose` elementwise (+) x2 x1

prop_int_div :: Gen Bool  -- Fails with arithmetic overflow for *_MIN or *_MAX ?
prop_int_div = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CChar, Tensor CChar)
  return $ x1 // x2 `equal` elementwise div x1 x2

prop_int_div_float :: Gen Bool
prop_int_div_float = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ x1 // x2 `equal` elementwise (\ a b -> fromIntegral $ floor $ a / b) x1 x2

prop_mod :: Gen Bool
prop_mod = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CChar, Tensor CChar)
  return $ x1 % x2 `equal` elementwise mod x1 x2

prop_mod_float :: Gen Bool
prop_mod_float = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CDouble, Tensor CDouble)
  return $ x1 % x2 `allClose` elementwise (\ a b -> a - b * fromIntegral (floor $ a / b)) x1 x2

prop_matmul_id :: Bool
prop_matmul_id = x @ x == x
  where
    x = eye 3 3 0 :: Tensor CFloat

prop_matmul_single :: Bool
prop_matmul_single = x @ x == 1
  where
    x = single 1 :: Tensor CFloat

prop_matmul_dot :: Bool
prop_matmul_dot = x @ x == 3
  where
    x = ones [3] :: Tensor CFloat

prop_matmul_matvec :: Bool
prop_matmul_matvec = a @ x == 3 * x
  where
    a = ones [3, 3] :: Tensor CFloat
    x = ones [3] :: Tensor CFloat

prop_matmul_unit_id :: Bool
prop_matmul_unit_id = unit_x @ eye last_dim last_dim 0 == unit_x
  where
    last_dim = last $ shape unit_x

prop_matmul_unit_matvec :: Bool
prop_matmul_unit_matvec = unit_x @ ones [last_dim] ==
  fromList2 [[ 21,  70, 119],
             [168, 217, 266]]
  where
    last_dim = last $ shape unit_x

prop_matmul_unit :: Bool
prop_matmul_unit = unit_x @ transpose unit_x ==
  fromList3 [[[   91,   238,   385],
              [  238,   728,  1218],
              [  385,  1218,  2051]],

             [[ 4060,  5236,  6412],
              [ 5236,  6755,  8274],
              [ 6412,  8274, 10136]]]

prop_getelem :: Index -> Bool
prop_getelem shape
  | totalElems shape /= 0 = zeros shape ! replicate (length shape) 0 == (0 :: CFloat)
  | otherwise             = True

prop_getelem_scalar ::  Bool
prop_getelem_scalar = scalar (0 :: CFloat) ! [] == 0

prop_getelem_single ::  Bool
prop_getelem_single = single (0 :: CFloat) ! [0] == 0

prop_numel :: Index -> Bool
prop_numel shape = numel x == V.length (tensorData x)
  where
    x = copy $ zeros shape :: Tensor CFloat

prop_min :: Tensor CDouble -> Bool
prop_min x
  | numel x /= 0 = T.min x == T.foldl' min (1 / 0) x
  | otherwise    = True

prop_max :: Tensor CDouble -> Bool
prop_max x
  | numel x /= 0 = T.max x == T.foldl' max (-1 / 0) x
  | otherwise    = True

prop_sum :: Index -> Bool
prop_sum shape = T.sum (ones shape :: Tensor CFloat) == fromIntegral (totalElems shape)

prop_sum_int :: Tensor CFloat -> Bool
prop_sum_int x = T.sum xI == T.foldl' (+) 0 xI
  where
    xI = astype x :: Tensor CLLong

prop_sum_distributive :: Index -> Gen Bool
prop_sum_distributive shape = do
  (x1, x2) <- arbitraryPairWithShape shape :: Gen (Tensor CDouble, Tensor CDouble)
  return $ scalar (T.sum x1 + T.sum x2) `allClose` scalar (T.sum (x1 + x2))

prop_sum_empty :: Index -> Bool
prop_sum_empty shape = T.sum (ones $ 0 : shape) == (0 :: CFloat)

prop_sum_scalar :: Bool
prop_sum_scalar = T.sum (scalar 1) == (1 :: CFloat)

prop_sum_single :: Bool
prop_sum_single = T.sum (single 1) == (1 :: CFloat)

prop_mean_empty :: Index -> Bool
prop_mean_empty shape = isNaN $ T.mean (ones $ 0 : shape :: Tensor CFloat)

prop_sum_along :: Tensor CFloat -> Bool
prop_sum_along x =
  sumAlongDim xI 0 `equal`
  tensor (tail $ shape xI) (\ index -> T.sum $ xI !: (A : map I index))
  where
    xI = insertDim x (-1)

prop_sum_along_ones :: Index -> Bool
prop_sum_along_ones shape =
  sumAlongDim (ones shapeI :: Tensor CFloat) 0 `equal`
  full (tail shapeI) (fromIntegral $ head shapeI)
  where
    shapeI = shape ++ [1]

prop_sum_along_distributive :: Index -> Gen Bool
prop_sum_along_distributive shape = do
  (x1, x2) <- arbitraryPairWithShape $ shape ++ [1] :: Gen (Tensor CDouble, Tensor CDouble)
  return $ sumAlongDim x1 0 + sumAlongDim x2 0 `allClose` sumAlongDim (x1 + x2) 0

prop_sum_along_empty :: Index -> Bool
prop_sum_along_empty shape =
  (sumAlongDim (zeros shapeI) 0 :: Tensor CFloat) `equal` zeros (tail shapeI)
  where
    shapeI = shape ++ [0]

prop_sum_along_single :: Bool
prop_sum_along_single = sumAlongDim (single 1 :: Tensor CFloat) 0 == 1

prop_sum_along_single_neg :: Bool
prop_sum_along_single_neg = sumAlongDim (single 1 :: Tensor CFloat) (-1) == 1

prop_sum_along_keep_dims :: Tensor CFloat -> Bool
prop_sum_along_keep_dims x =
  sumAlongDims xI [0] True ==
  tensor (1 : tail (shape xI)) (\ index -> T.sum $ xI !: (A : map I (tail index)))
  where
    xI = insertDim x (-1)

prop_sum_along_keep_dims_neg :: Tensor CFloat -> Bool
prop_sum_along_keep_dims_neg x =
  sumAlongDims xI [-1] True ==
  tensor (init (shape xI) ++ [1]) (\ index -> T.sum $ xI !: (map I (init index) ++ [A]))
  where
    xI = insertDim x 0

prop_sum_along_all :: Tensor CFloat -> Bool
prop_sum_along_all x =
  sumAlongDims x [0 .. dim x - 1] False == scalar (T.sum x)

prop_sum_along_keep_dims_all :: Tensor CFloat -> Bool
prop_sum_along_keep_dims_all x =
  sumAlongDims x [0 .. dim x - 1] True == full (replicate (dim x) 1) (T.sum x)

prop_copy :: Tensor CFloat -> Bool
prop_copy x = x `equal` copy x

prop_copy_offset :: Tensor CFloat -> Bool
prop_copy_offset x = tensorOffset (copy x) == 0

prop_copy_numel :: Tensor CFloat -> Bool
prop_copy_numel x = numel x1 == V.length (tensorData x1)
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

prop_flatten_view :: Tensor CFloat -> Bool
prop_flatten_view x = x `view` [-1] `equal` flatten x

prop_view :: Tensor CFloat -> Bool
prop_view x =
  transpose (
    (xT `view` [totalElems $ shape x]) `view` shape xT
  ) `equal` x1
  where
    x1 = insertDim (insertDim x 0) 0
    xT = transpose x1

prop_view_neg1 :: Tensor CFloat -> Bool
prop_view_neg1 x =
  (x `view` [-1]) `view` shape x `equal` x

prop_view_neg2 :: Tensor CFloat -> Bool
prop_view_neg2 x
  | numel x /= 0 = (x `view` (-1 : shape x)) `view` shape x `equal` x
  | otherwise    = True

prop_view_neg3 :: Tensor CFloat -> Bool
prop_view_neg3 x
  | numel x /= 0 = (x `view` (shape x ++ [-1])) `view` shape x `equal` x
  | otherwise    = True

prop_insert_dim :: Index -> Bool
prop_insert_dim shape = insertDim x 0 `equal` zeros (1 : shape)
  where
    x = zeros shape :: Tensor CFloat

prop_insert_dim_neg :: Index -> Bool
prop_insert_dim_neg shape = insertDim x (-1) `equal` zeros (shape ++ [1])
  where
    x = zeros shape :: Tensor CFloat

prop_insert_dim_scalar :: Bool
prop_insert_dim_scalar = insertDim (scalar 0 :: Tensor CFloat) 0 `equal` single 0

prop_insert_dim_scalar_neg :: Bool
prop_insert_dim_scalar_neg = insertDim (scalar 0 :: Tensor CFloat) (-1) `equal` single 0

prop_insert_dim_single0 :: Bool
prop_insert_dim_single0 = insertDim (single 0 :: Tensor CFloat) 0 `equal` zeros [1, 1]

prop_insert_dim_single1 :: Bool
prop_insert_dim_single1 = insertDim (single 0 :: Tensor CFloat) 1 `equal` zeros [1, 1]

prop_insert_dim_single_neg :: Bool
prop_insert_dim_single_neg = insertDim (single 0 :: Tensor CFloat) (-1) `equal` zeros [1, 1]

prop_swap_dims_id1 :: Tensor CFloat -> Bool
prop_swap_dims_id1 x = swapDims xI (0, 0) `equal` xI
  where
    xI = insertDim x (-1)

prop_swap_dims_id2 :: Tensor CFloat -> Bool
prop_swap_dims_id2 x = xI `swapDims` (0, 1) `swapDims` (0, 1) `equal` xI
  where
    xI = insertDims x [-1, -2]


-- Boolean
-- -------

prop_equal :: Tensor CFloat -> Bool
prop_equal x = x `equal` x

prop_not_equal :: Tensor CFloat -> Bool
prop_not_equal x
  | numel x /= 0 = not $ x `equal` (x + 1)
  | otherwise             = True

prop_allclose :: Tensor CFloat -> Bool
prop_allclose x = allClose x x

prop_not_allclose :: Tensor CFloat -> Bool
prop_not_allclose x
  | numel x /= 0 = not $ allClose x (x + 1)
  | otherwise             = True

prop_tensor_equal1 :: Tensor CFloat -> Bool
prop_tensor_equal1 x = (x T.== x) `equal` onesLike x

prop_tensor_equal0 :: Tensor CFloat -> Bool
prop_tensor_equal0 x
  | numel x /= 0 = (x T.== x + 1) `equal` zerosLike x
  | otherwise             = True

prop_tensor_not_equal0 :: Tensor CFloat -> Bool
prop_tensor_not_equal0 x = (x T./= x) `equal` zerosLike x

prop_tensor_not_equal1 :: Tensor CFloat -> Bool
prop_tensor_not_equal1 x
  | numel x /= 0 = (x T./= x + 1) `equal` onesLike x
  | otherwise             = True

prop_tensor_less :: Tensor CFloat -> Bool
prop_tensor_less x = (x T.< x) `equal` zerosLike x

prop_tensor_leq :: Tensor CFloat -> Bool
prop_tensor_leq x = (x T.<= x) `equal` onesLike x

prop_tensor_not_greater :: Gen Bool
prop_tensor_not_greater = do
  (x1, x2) <- arbitraryBroadcastablePair :: Gen (Tensor CFloat, Tensor CFloat)
  return $ T.not (x1 T.> x2) `equal` (x1 T.<= x2)

prop_not :: Index -> Bool
prop_not shape = T.not (zeros shape) `equal` (ones shape :: Tensor CBool)

prop_and :: Index -> Bool
prop_and shape = (ones shape & ones shape) `equal` (ones shape :: Tensor CBool) &&
                 (zeros shape & ones shape) `equal` (zeros shape :: Tensor CBool) &&
                 (zeros shape & zeros shape) `equal` (zeros shape :: Tensor CBool)

prop_or :: Index -> Bool
prop_or shape = (ones shape |. ones shape) `equal` (ones shape :: Tensor CBool) &&
                (zeros shape |. ones shape) `equal` (ones shape :: Tensor CBool) &&
                (zeros shape |. zeros shape) `equal` (zeros shape :: Tensor CBool)


-- NumTensor
-- ---------

prop_abs :: Index -> Bool
prop_abs shape = abs x `equal` x
  where
    x = ones shape :: Tensor CFloat

prop_abs_neg :: Index -> Bool
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

prop_num_associative :: Index -> Gen Bool
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
  show (zeros [1, 2, 0] :: Tensor CFloat) ==
  "tensor([], shape=[1,2,0], dtype=float32)"


-- AdvancedIndex
-- -------------

prop_slice_index :: Tensor CFloat -> Bool
prop_slice_index x
  | numel x /= 0 = scalar (x ! replicate (dim x) 0) `equal` x !: replicate (dim x) 0
  | otherwise    = True

prop_slice_single :: Tensor CFloat -> Bool
prop_slice_single x
  | numel x /= 0 = x ! replicate (dim x) 0 == item (x !: replicate (dim x) (0:.1))
  | otherwise    = True

prop_slice_all :: Tensor CFloat -> Bool
prop_slice_all x = x !: replicate (dim x) A `equal` x

prop_slice_all0 :: Tensor CFloat -> Bool
prop_slice_all0 x = x !: [] `equal` x

prop_slice_all1 :: Tensor CFloat -> Bool
prop_slice_all1 x
  | dim x > 0 = x !: [A] `equal` x
  | otherwise = True

prop_slice_start_all :: Tensor CFloat -> Bool
prop_slice_start_all x = x !: replicate (dim x) (S 0) `equal` x

prop_slice_start1_all :: Tensor CFloat -> Bool
prop_slice_start1_all x
  | dim x > 0 = x !: [S 0] `equal` x
  | otherwise = True

prop_slice_start_equiv :: Tensor CFloat -> Bool
prop_slice_start_equiv x
  | dim x > 0 = x !: [S 1] `equal` x !: [1 :. head (shape x)]
  | otherwise = True

prop_slice_end_all :: Tensor CFloat -> Bool
prop_slice_end_all x = x !: map E (shape x) `equal` x

prop_slice_end1_all :: Tensor CFloat -> Bool
prop_slice_end1_all x
  | dim x > 0 = x !: [E $ head $ shape x] `equal` x
  | otherwise = True

prop_slice_end_equiv :: Tensor CFloat -> Bool
prop_slice_end_equiv x
  | dim x > 0 = x !: [E (-1)] `equal` x !: [0 :. head (shape x) - 1]
  | otherwise = True

prop_slice_neg :: Tensor CFloat -> Bool
prop_slice_neg x
  | dim x > 0 =
    x !: [-3:. -1] `equal` x !: [I (max 0 (head (shape x) - 3)) :. head (shape x) - 1]
  | otherwise = True

prop_slice_none :: Tensor CFloat -> Bool
prop_slice_none x = insertDim x 0 `equal` x !: [None]

prop_slice_ell :: Tensor CFloat -> Bool
prop_slice_ell x
  | dim x > 0 =
    x !: [Ell, 0:.1] `equal`
    x !: (replicate (dim x - 1) A ++ [0 :. 1])
  | otherwise =
    True

prop_slice_insert_dim :: Tensor CFloat -> Bool
prop_slice_insert_dim x = insertDim x (-1) `equal` x !: [Ell, None]

prop_slice_step1 :: Tensor CFloat -> Bool
prop_slice_step1 x = x !: replicate (dim x) (A :. 1) `equal` x

prop_slice_step11 :: Tensor CFloat -> Bool
prop_slice_step11 x
  | dim x > 0 = x !: [A :. 1] `equal` x
  | otherwise          = True

prop_slice_neg_step_id :: Tensor CFloat -> Bool
prop_slice_neg_step_id x = x !: slice !: slice `equal` x
  where
    slice = replicate (dim x) (A :. -1)

prop_slice_neg_step_id1 :: Tensor CFloat -> Bool
prop_slice_neg_step_id1 x
  | dim x > 0 = x !: [A :. -1] !: [A :. -1] `equal` x
  | otherwise = True

prop_slice_step :: Bool
prop_slice_step = (arange 0 10 1 :: Tensor CFloat) !: [A :. 2] `equal` arange 0 10 2

prop_slice_neg_step :: Bool
prop_slice_neg_step = (arange 0 10 1 :: Tensor CFloat) !: [A :. -2] `allClose` arange 9 (-1) (-2)

prop_itensor_index :: Tensor CFloat -> Bool
prop_itensor_index x
  | numel x /= 0 = scalar (x ! replicate (dim x) 0) `equal` x !: replicate (dim x) (T $ scalar 0)
  | otherwise    = True

prop_itensor_single :: Tensor CFloat -> Bool
prop_itensor_single x
  | numel x /= 0 = x ! replicate (dim x) 0 == item (x !: replicate (dim x) (T $ single 0))
  | otherwise    = True

-- prop_itensor_all :: Tensor CFloat -> Bool
-- prop_itensor_all x = x !: split (indices $ shape x) 0 == x

unit_x :: Tensor CFloat
unit_x = arange 0 (2 * 3 * 7) 1 `view` [2, 3, 7]

prop_slice_unit1 :: Bool
prop_slice_unit1 = unit_x !: [A, None, 0, None, A, None] ==
  fromList5 [[[[[ 0],
                [ 1],
                [ 2],
                [ 3],
                [ 4],
                [ 5],
                [ 6]]]],



            [[[[21],
                [22],
                [23],
                [24],
                [25],
                [26],
                [27]]]]]

prop_slice_unit2 :: Bool
prop_slice_unit2 = unit_x !: [None, Ell, None] ==
  fromList5 [[[[[ 0],
                [ 1],
                [ 2],
                [ 3],
                [ 4],
                [ 5],
                [ 6]],

              [[ 7],
                [ 8],
                [ 9],
                [10],
                [11],
                [12],
                [13]],

              [[14],
                [15],
                [16],
                [17],
                [18],
                [19],
                [20]]],


              [[[21],
                [22],
                [23],
                [24],
                [25],
                [26],
                [27]],

              [[28],
                [29],
                [30],
                [31],
                [32],
                [33],
                [34]],

              [[35],
                [36],
                [37],
                [38],
                [39],
                [40],
                [41]]]]]

prop_slice_unit3 :: Bool
prop_slice_unit3 = unit_x !: [Ell, None, 2 :. 9] ==
  fromList4 [[[[ 2,  3,  4,  5,  6]],

              [[ 9, 10, 11, 12, 13]],

              [[16, 17, 18, 19, 20]]],


            [[[23, 24, 25, 26, 27]],

              [[30, 31, 32, 33, 34]],

              [[37, 38, 39, 40, 41]]]]

prop_slice_unit4 :: Bool
prop_slice_unit4 = unit_x !: [Ell, 5 :. 9, None] ==
  fromList4 [[[[ 5],
               [ 6]],

              [[12],
               [13]],

              [[19],
               [20]]],


             [[[26],
               [27]],

              [[33],
               [34]],

              [[40],
               [41]]]]

prop_slice_unit5 :: Bool
prop_slice_unit5 = unit_x !: [Ell, -7 :. -2 :. -1, None] == zeros [2, 3, 0, 1]

prop_itensor_unit1 :: Bool
prop_itensor_unit1 = unit_x !: [A, 0, T $ fromList [1, 2, 3]] ==
  fromList2 [[ 1,  2,  3],
             [22, 23, 24]]

prop_itensor_unit2 :: Bool
prop_itensor_unit2 = unit_x !: [None, Ell, T $ fromList [0, -1, 3]] ==
  fromList4 [[[[ 0,  6,  3],
               [ 7, 13, 10],
               [14, 20, 17]],

              [[21, 27, 24],
               [28, 34, 31],
               [35, 41, 38]]]]

prop_itensor_unit3 :: Bool
prop_itensor_unit3 = unit_x !: [None, T $ fromList2 [[0, -1, -1],
                                                     [1,  1,  0]],
                                      T $ fromList [0, -1, -2], 0] ==
  fromList3 [[[ 0, 35, 28],
              [21, 35,  7]]]

prop_itensor_unit4 :: Bool
prop_itensor_unit4 = unit_x !: [T $ fromList2 [[0, -1, -1],
                                               [1,  1,  0]],
                                S 1, T $ fromList [0, -1, -2]] ==
  fromList3 [[[ 7, 14],
              [34, 41],
              [33, 40]],

            [[28, 35],
              [34, 41],
              [12, 19]]]

prop_itensor_unit5 :: Bool
prop_itensor_unit5 = unit_x !: [T $ fromList2 [[0, -1, -1],
                                               [1,  1,  0]],
                                1, T $ fromList [0, -1, -2]] ==
  fromList2 [[ 7, 34, 33],
             [28, 34, 12]]

return []

main :: IO ()
main = do
  success <- $quickCheckAll
  -- success <- $forAllProperties $ quickCheckWithResult (stdArgs {maxSuccess = 10000})
  unless success exitFailure
