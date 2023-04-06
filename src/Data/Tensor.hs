{-# LANGUAGE InstanceSigs #-}

module Data.Tensor where

import Control.Applicative
import Data.Vector (generate, fromList, Vector, (!), (//))
import qualified Data.Vector as V
import Data.Matrix (matrix, elementwiseUnsafe, Matrix, (!))
import qualified Data.Matrix as M
import System.Random
import Data.Random.Normal
import Data.Index
import Foreign
-- import System.IO.Unsafe

data Tensor t = Tensor {
  shape :: Index,
  tensorData :: Vector (Matrix t)
} deriving Show

type TensorIndex = [Tensor Int]

instance Functor Tensor where
  fmap :: (a -> b) -> Tensor a -> Tensor b
  fmap f (Tensor shape dat) =
    Tensor shape $ fmap (fmap f) dat

instance Applicative Tensor where
  pure :: a -> Tensor a
  pure elem = tensor [1] $ const elem

  liftA2 :: (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
  liftA2 f (Tensor shape dat1) (Tensor _ dat2) =
    Tensor shape $ V.zipWith (elementwiseUnsafe f) dat1 dat2

instance Foldable Tensor where
  foldMap :: Monoid m => (a -> m) -> Tensor a -> m
  foldMap f x@(Tensor _ dat) =
    foldMap (foldMap f) dat

  foldr :: (a -> b -> b) -> b -> Tensor a -> b
  foldr f accum x@(Tensor _ dat) =
    foldr (flip $ foldr f) accum dat

  sum :: Num t => Tensor t -> t
  sum = uncurry (+) . foldr kahanSum (0, 0)
    where
      kahanSum item (sum, c) =
        let y = item - c
            t = sum + y in
          (t, (t - sum) - y)

instance (Num t) => Num (Tensor t) where
  (+) :: Num t => Tensor t -> Tensor t -> Tensor t
  (+) = performWithBroadcasting (+)

  (-) :: Num t => Tensor t -> Tensor t -> Tensor t
  (-) = performWithBroadcasting (-)

  (*) :: Num t => Tensor t -> Tensor t -> Tensor t
  (*) = performWithBroadcasting (*)

  abs :: Num t => Tensor t -> Tensor t
  abs = fmap abs

  signum :: Num t => Tensor t -> Tensor t
  signum = fmap signum

  fromInteger :: Num t => Integer -> Tensor t
  fromInteger = pure . fromInteger

instance (Fractional t) => Fractional (Tensor t) where
  (/) :: Fractional t => Tensor t -> Tensor t -> Tensor t
  (/) = performWithBroadcasting (/)

  fromRational :: Fractional t => Rational -> Tensor t
  fromRational = pure . fromRational

instance (Floating t) => Floating (Tensor t) where
  pi :: Floating t => Tensor t
  pi = pure pi
  exp :: Floating t => Tensor t -> Tensor t
  exp = fmap exp
  log :: Floating t => Tensor t -> Tensor t
  log = fmap log
  sin :: Floating t => Tensor t -> Tensor t
  sin = fmap sin
  cos :: Floating t => Tensor t -> Tensor t
  cos = fmap cos
  asin :: Floating t => Tensor t -> Tensor t
  asin = fmap asin
  acos :: Floating t => Tensor t -> Tensor t
  acos = fmap acos
  atan :: Floating t => Tensor t -> Tensor t
  atan = fmap atan
  sinh :: Floating t => Tensor t -> Tensor t
  sinh = fmap sinh
  cosh :: Floating t => Tensor t -> Tensor t
  cosh = fmap cosh
  asinh :: Floating t => Tensor t -> Tensor t
  asinh = fmap asinh
  acosh :: Floating t => Tensor t -> Tensor t
  acosh = fmap acosh
  atanh :: Floating t => Tensor t -> Tensor t
  atanh = fmap atanh


tensor :: Index -> (Index -> t) -> Tensor t
tensor shape builder =
  Tensor shape
  $ generate (countIndex arrayShape) (
    \ i ->
      let index = fromInt arrayShape i in
        matrix matrixRows matrixCols
        $ builder
        . mergeIndex index
  )
  where
    (arrayShape, (matrixRows, matrixCols)) = splitIndex shape

full :: Index -> t -> Tensor t
full shape value = tensor shape $ const value

zeros :: Num t => Index -> Tensor t
zeros shape = full shape 0

ones :: Num t => Index -> Tensor t
ones shape = full shape 1

eye :: Num t => Index -> Tensor t
eye shape = tensor shape $ fromBool . allEqual

randn :: (RandomGen g, Random t, Floating t) => Index -> g -> (Tensor t, g)
randn shape gen =
  (tensor shape (\ index -> randomVector V.! toInt shape index), newGen)
  where
    n = countIndex shape
    (randomList, newGen) = go n [] gen
    go 0 xs g = (xs, g)
    go count xs g =
      let (randomValue, newG) = normal g in
        go (count - 1) (randomValue : xs) newG
    randomVector = fromList randomList

-- | low -> high (inclusive)
arange :: Num t => Int -> Int -> Tensor t
arange low high =
  tensor [high - low + 1] (
    \ index ->
      fromIntegral $ head index + low - 1
  )


dot :: Num t => Tensor t -> Tensor t -> Tensor t
dot (Tensor shape1 dat1) (Tensor shape2 dat2) =
  Tensor (mergeIndex arrayShape (matrixRows1, matrixCols2))
  $ liftA2 (*) dat1 dat2
  where
    (arrayShape, (matrixRows1, _)) = splitIndex shape1
    (_, (_, matrixCols2)) = splitIndex shape2

-- | An infix synonym for dot.
(@) :: Num t => Tensor t -> Tensor t -> Tensor t
(@) = dot


getElem :: Tensor t -> Index -> t
getElem (Tensor shape dat) index
  | validateIndex shape nIndex =
    (dat V.! arrayIndex) M.! matrixIndex
  | otherwise =
    error
    $ "incorrect index "
    ++ show index
    ++ " for shape "
    ++ show shape
  where
    nIndex = normalizeIndex shape index
    (arrayIndex, matrixIndex) = toInternal shape nIndex

(!?) :: Tensor t -> Index -> t
(!?) = getElem

slice :: Tensor t -> Slices -> Tensor t
slice x@(Tensor shape _) slices =
  tensor newShape (\ index -> x !? zipWith (!*) expandedIndex index)
  where
    expandedIndex =
      zipWith (
        \ dim slice ->
          if null slice then
            [1 .. dim]
          else slice
      ) shape slices
    newShape = map length expandedIndex

(!:) :: Tensor t -> Slices -> Tensor t
(!:) = slice

validateTensorIndex :: TensorIndex -> Bool
validateTensorIndex = allEqual . map shape

advancedIndex :: Tensor t -> TensorIndex -> Tensor t
advancedIndex x tensorIndex
  | validateTensorIndex tensorIndex =
    tensor (shape $ head tensorIndex) (
      \ index ->
        x !? map (!? index) tensorIndex
    )
  | otherwise =
    error
    $ "incorrect index "
    ++ show tensorIndex
    ++ " for shape "
    ++ show (shape x)

(!.) :: Tensor t -> TensorIndex -> Tensor t
(!.) = advancedIndex


numel :: Tensor t -> Int
numel (Tensor shape _) = countIndex shape

item :: Tensor t -> t
item x = x !? [1]

transpose :: Tensor t -> Tensor t
transpose (Tensor shape dat) =
  Tensor shape $ fmap M.transpose dat

flatten :: Tensor t -> Tensor t
flatten x@(Tensor shape _) =
  tensor [numel x] $ (x !?) . fromInt shape . head

sumBabushka :: (Num t, Ord t) => Tensor t -> t
sumBabushka = uncurry (+) . foldr kahanBabushkaSum (0, 0)
  where
    kahanBabushkaSum item (sum, c) =
      if abs sum >= abs item then
        let y = item - c
            t = sum + y in
          (t, (t - sum) - y)
      else
        let y = sum - c
            t = item + y in
          (t, (t - item) - y)

mean :: Fractional t => Tensor t -> t
mean x = sum x / fromIntegral (numel x)

sumAlongDim :: Num t => Tensor t -> Int -> Tensor t
sumAlongDim x@(Tensor shape _) dim =
  tensor newShape (
    \ index ->
      let slices = map pure index in
        sum . slice x
        $ take (nDim - 1) slices
        ++ [[1 .. (shape !* normalizeItem (length shape) nDim)]]
        ++ drop (nDim - 1) slices
  )
  where
    nDim = normalizeItem (length shape) dim
    newShape = take (nDim - 1) shape ++ drop nDim shape

sumAlongDimKeepDims :: Num t => Tensor t -> Int -> Tensor t
sumAlongDimKeepDims x@(Tensor shape _) dim =
  tensor newShape (
    \ index ->
      let slices = map pure index in
        sum . slice x
        $ take (nDim - 1) slices
        ++ [[1 .. (shape !* nDim)]]
        ++ drop nDim slices
  )
  where
    nDim = normalizeItem (length shape) dim
    newShape = take (nDim - 1) shape ++ [1] ++ drop nDim shape

insertDim :: Tensor t -> Int -> Tensor t
insertDim x@(Tensor shape _) dim =
  tensor newShape (
    \ index ->
        x !? (take (nDim - 1) index ++ drop nDim index)
  )
  where
    nDim = normalizeItem (length shape) dim
    newShape = take nDim shape ++ [1] ++ drop nDim shape

-- | tensor -> dim -> times
repeatAlongDim :: Tensor t -> Int -> Int -> Tensor t
repeatAlongDim x@(Tensor shape _) dim times =
  tensor newShape (
    \ index ->
        x !? (
          take (nDim - 1) index
          ++ [(((index !* nDim) - 1) `mod` currentDim) + 1]
          ++ drop nDim index
        )
  )
  where
    nDim = normalizeItem (length shape) dim
    currentDim = shape !* nDim
    newShape =
      take (nDim - 1) shape
      ++ [currentDim * times]
      ++ drop nDim shape
    -- debugPrint x =
    --   unsafePerformIO $ print x >> return x

-- | tensor -> dims -> times
repeatAlongDims :: Tensor t -> [Int] -> [Int] -> Tensor t
repeatAlongDims x dims timess =
  foldr (
    \(dim, times) accum ->
      repeatAlongDim accum dim times
  ) x $ zip dims timess

swapDim :: Tensor t -> Int -> Int -> Tensor t
swapDim x@(Tensor shape _) from to =
  tensor (swapElementsAt from to shape) ((x !?) . swapElementsAt from to)

performWithBroadcasting ::
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
performWithBroadcasting f x1@(Tensor shape1 _) x2@(Tensor shape2 _)
  | shape1 == [1] = fmap (f $ item x1) x2
  | shape2 == [1] = fmap (flip f $ item x2) x1
  | otherwise     = uncurry (liftA2 f) $ broadcast x1 x2

verifyBroadcastable :: Tensor a -> Tensor b -> Bool
verifyBroadcastable (Tensor shape1 _) (Tensor shape2 _) =
  length shape1 == length shape2
  && and (
    zipWith (
      \ dim1 dim2 -> dim1 == dim2 || dim1 == 1 || dim2 == 1
    ) shape1 shape2
  )

broadcast :: Tensor a -> Tensor b -> (Tensor a, Tensor b)
broadcast x1@(Tensor shape1 _) x2@(Tensor shape2 _)
  | verifyBroadcastable x1 x2 =
    (repeatAlongDims x1 dims1 times1,
     repeatAlongDims x2 dims2 times2)
  | otherwise =
    error
    $ "tensors "
    ++ show shape1
    ++ " "
    ++ show shape2
    ++ " can not be broadcasted"
  where
    zipped = zip3 [1 .. length shape1] shape1 shape2
    (dims1, times1) = foldr addDim ([], []) zipped
    (dims2, times2) = foldr (addDim . flipDims) ([], []) zipped
    addDim (i, dim1, dim2) accum@(accumDims, accumTimes) =
      if dim1 /= dim2 && dim1 == 1 then
        (i : accumDims, dim2 : accumTimes)
      else accum
    flipDims (i, dim1, dim2) = (i, dim2, dim1)
