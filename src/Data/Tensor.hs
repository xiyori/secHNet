{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
-- {-# LANGUAGE BangPatterns #-}

module Data.Tensor where

import Control.Applicative
import Data.MonoTraversable
import Data.Vector.Unboxed (generate, fromList, toList, singleton,
                            Unbox, Vector, (!), (//))
import qualified Data.Vector.Unboxed as V
import Data.Matrix (matrix, Matrix)
import qualified Data.Matrix as M
import Data.List (foldl')
import System.Random
import Data.Random.Normal
import Data.Index
import Foreign
-- import System.IO.Unsafe

data (Unbox t) =>
  Tensor t = Tensor {
    shape :: !Index,
    tensorData :: !(Vector t)
  } deriving Show

type instance Element (Tensor t) = t

type TensorIndex = [Tensor Int]

instance Unbox t => MonoFunctor (Tensor t) where
  omap :: (t -> t) -> Tensor t -> Tensor t
  omap f (Tensor shape dat) = Tensor shape $ V.map f dat
  {-# INLINE omap #-}

instance Unbox t => MonoFoldable (Tensor t) where
  ofoldMap :: (Monoid m) => (t -> m) -> Tensor t -> m
  ofoldMap f (Tensor _ dat) = V.foldMap f dat
  ofoldr :: (t -> b -> b) -> b -> Tensor t -> b
  ofoldr f accum (Tensor _ dat) = V.foldr f accum dat
  ofoldl' :: (a -> t -> a) -> a -> Tensor t -> a
  ofoldl' f accum (Tensor _ dat) = V.foldl' f accum dat
  ofoldr1Ex :: (t -> t -> t) -> Tensor t -> t
  ofoldr1Ex f (Tensor _ dat) = V.foldr1 f dat
  ofoldl1Ex' :: (t -> t -> t) -> Tensor t -> t
  ofoldl1Ex' f (Tensor _ dat) = V.foldr1' f dat
  {-# INLINE ofoldMap #-}
  {-# INLINE ofoldr #-}
  {-# INLINE ofoldl' #-}
  {-# INLINE ofoldr1Ex #-}
  {-# INLINE ofoldl1Ex' #-}

instance (Unbox t, Num t) => Num (Tensor t) where
  (+) = performWithBroadcasting (+)
  (-) = performWithBroadcasting (-)
  (*) = performWithBroadcasting (*)
  abs = omap abs
  signum = omap signum
  fromInteger = single . fromInteger
  {-# INLINE (+) #-}
  {-# INLINE (-) #-}
  {-# INLINE (*) #-}
  {-# INLINE abs #-}
  {-# INLINE signum #-}
  {-# INLINE fromInteger #-}

instance (Unbox t, Fractional t) => Fractional (Tensor t) where
  (/) = performWithBroadcasting (/)
  fromRational = single . fromRational
  {-# INLINE (/) #-}
  {-# INLINE fromRational #-}

instance (Unbox t, Floating t) => Floating (Tensor t) where
  pi = single pi
  exp = omap exp
  log = omap log
  sin = omap sin
  cos = omap cos
  asin = omap asin
  acos = omap acos
  atan = omap atan
  sinh = omap sinh
  cosh = omap cosh
  asinh = omap asinh
  acosh = omap acosh
  atanh = omap atanh
  {-# INLINE pi #-}
  {-# INLINE exp #-}
  {-# INLINE log #-}
  {-# INLINE sin #-}
  {-# INLINE cos #-}
  {-# INLINE asin #-}
  {-# INLINE acos #-}
  {-# INLINE atan #-}
  {-# INLINE sinh #-}
  {-# INLINE cosh #-}
  {-# INLINE asinh #-}
  {-# INLINE acosh #-}
  {-# INLINE atanh #-}


tensor :: (Unbox t) => Index -> (Index -> t) -> Tensor t
tensor shape builder =
  Tensor shape
  $ generate (countIndex shape)
  $ builder . unravelIndex shape

full :: (Unbox t) => Index -> t -> Tensor t
full shape value =
  Tensor shape $ V.replicate (countIndex shape) value

zeros :: (Unbox t, Num t) => Index -> Tensor t
zeros shape = full shape 0

ones :: (Unbox t, Num t) => Index -> Tensor t
ones shape = full shape 1

single :: (Unbox t) => t -> Tensor t
single = full $ singleton 1

eye :: (Unbox t, Num t) => Index -> Tensor t
eye shape = tensor shape $ fromBool . allEqualV

randn :: (Unbox t, RandomGen g, Random t, Floating t) =>
  Index -> g -> (Tensor t, g)
randn shape gen =
  (Tensor shape randomVector, newGen)
  where
    n = countIndex shape
    (randomList, newGen) = go n [] gen
    go 0 xs g = (xs, g)
    go count xs g =
      let (randomValue, newG) = normal g in
        go (count - 1) (randomValue : xs) newG
    randomVector = fromList randomList

-- | low -> high (inclusive)
arange :: (Unbox t, Num t) => Int -> Int -> Tensor t
arange low high =
  tensor (singleton $ high - low + 1) (
    \ index ->
      fromIntegral $ V.head index + low - 1
  )


map :: (Unbox a, Unbox b) => (a -> b) -> Tensor a -> Tensor b
map f (Tensor shape dat) = Tensor shape $ V.map f dat

elementwise :: (Unbox a, Unbox b, Unbox c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
elementwise f (Tensor shape dat1) (Tensor _ dat2) =
  Tensor shape $ V.zipWith f dat1 dat2

performWithBroadcasting :: (Unbox a, Unbox b, Unbox c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
performWithBroadcasting f x1@(Tensor shape1 _) x2@(Tensor shape2 _)
  | V.head shape1 == 1 && V.length shape1 == 1 =
    Data.Tensor.map (f $ item x1) x2
  | V.head shape2 == 1 && V.length shape2 == 1 =
    Data.Tensor.map (flip f $ item x2) x1
  | otherwise =
    uncurry (elementwise f) $ broadcast x1 x2

verifyBroadcastable :: (Unbox a, Unbox b) =>
  Tensor a -> Tensor b -> Bool
verifyBroadcastable (Tensor shape1 _) (Tensor shape2 _) =
  V.length shape1 == V.length shape2
  && V.and (
    V.zipWith (
      \ dim1 dim2 -> dim1 == dim2 || dim1 == 1 || dim2 == 1
    ) shape1 shape2
  )

broadcast :: (Unbox a, Unbox b) =>
  Tensor a -> Tensor b -> (Tensor a, Tensor b)
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
    zipped = V.zip3 (fromList [0 .. V.length shape1 - 1]) shape1 shape2
    (dims1, times1) = V.foldr addDim ([], []) zipped
    (dims2, times2) = V.foldr (addDim . flipDims) ([], []) zipped
    addDim (i, dim1, dim2) accum@(accumDims, accumTimes) =
      if dim1 /= dim2 && dim1 == 1 then
        (i : accumDims, dim2 : accumTimes)
      else accum
    flipDims (i, dim1, dim2) = (i, dim2, dim1)

-- dot :: Num t => Tensor t -> Tensor t -> Tensor t
-- dot (Tensor shape1 dat1) (Tensor shape2 dat2) =
--   Tensor (mergeIndex arrayShape (matrixRows1, matrixCols2))
--   $ liftA2 (*) dat1 dat2
--   where
--     (arrayShape, (matrixRows1, _)) = splitIndex shape1
--     (_, (_, matrixCols2)) = splitIndex shape2

-- | An infix synonym for dot.
-- (@) :: Num t => Tensor t -> Tensor t -> Tensor t
-- (@) = dot


getElem :: (Unbox t) => Tensor t -> Index -> t
getElem (Tensor shape dat) index
  | validateIndex shape nIndex =
    dat V.! i
  | otherwise =
    error
    $ "incorrect index "
    ++ show index
    ++ " for shape "
    ++ show shape
  where
    nIndex = normalizeIndex shape index
    i = flattenIndex shape nIndex

(!?) :: (Unbox t) => Tensor t -> Index -> t
(!?) = getElem

-- | TODO: numpy-style slices with custom data Slice
slice :: (Unbox t) => Tensor t -> Slices -> Tensor t
slice x@(Tensor shape _) slices =
  tensor newShape (
    \ index ->
      x !? fromList (zipWith (!) expandedIndex $ toList index)
  )
  where
    expandedIndex =
      zipWith (
        \ dim slice ->
          if V.null slice then
            fromList [0 .. dim - 1]
          else slice
      ) (toList shape) slices
    newShape = fromList $ Prelude.map V.length expandedIndex

(!:) :: (Unbox t) => Tensor t -> Slices -> Tensor t
(!:) = slice

validateTensorIndex :: TensorIndex -> Bool
validateTensorIndex = allEqual . Prelude.map shape

advancedIndex :: (Unbox t) => Tensor t -> TensorIndex -> Tensor t
advancedIndex x tensorIndex
  | validateTensorIndex tensorIndex =
    tensor (shape $ head tensorIndex) (
      \ index ->
        x !? fromList (Prelude.map (!? index) tensorIndex)
    )
  | otherwise =
    error
    $ "incorrect index "
    ++ show tensorIndex
    ++ " for shape "
    ++ show (shape x)

(!.) :: (Unbox t) => Tensor t -> TensorIndex -> Tensor t
(!.) = advancedIndex


numel :: (Unbox t) => Tensor t -> Int
numel (Tensor shape _) = countIndex shape

item :: (Unbox t) => Tensor t -> t
item x = x !? singleton 1

transpose :: (Unbox t) => Tensor t -> Tensor t
transpose x = swapDim x (-1) (-2)

flatten :: (Unbox t) => Tensor t -> Tensor t
flatten x@(Tensor _ dat) = Tensor (singleton $ numel x) dat

sum :: (Unbox t, Num t) => Tensor t -> t
sum = fst . ofoldl' kahanSum (0, 0)
    where
      kahanSum (sum, c) item =
        let y = item - c
            t = sum + y in
          (t, (t - sum) - y)

sumBabushka :: (Unbox t, Num t, Ord t) => Tensor t -> t
sumBabushka = fst . ofoldl' kahanBabushkaSum (0, 0)
  where
    kahanBabushkaSum (sum, c) item =
      if abs sum >= abs item then
        let y = item - c
            t = sum + y in
          (t, (t - sum) - y)
      else
        let y = sum - c
            t = item + y in
          (t, (t - item) - y)

mean :: (Unbox t, Fractional t) => Tensor t -> t
mean x = osum x / fromIntegral (numel x)

sumAlongDim :: (Unbox t, Num t) =>
  Tensor t -> Int -> Tensor t
sumAlongDim x@(Tensor shape _) dim =
  tensor newShape (
    \ index ->
      let slices = Prelude.map singleton $ toList index in
        Data.Tensor.sum . slice x
        $ concat [
          take nDim slices,
          [fromList [0 .. (shape ! nDim) - 1]],
          drop nDim slices
        ]
  )
  where
    nDim = normalizeItem (V.length shape) dim
    newShape = V.concat [
        V.take nDim shape,
        V.drop (nDim + 1) shape
      ]

sumAlongDimKeepDims :: (Unbox t, Num t) =>
  Tensor t -> Int -> Tensor t
sumAlongDimKeepDims x@(Tensor shape _) dim =
  tensor newShape (
    \ index ->
      let slices = Prelude.map singleton $ toList index in
        Data.Tensor.sum . slice x
        $ concat [
          take nDim slices,
          [fromList [0 .. (shape ! nDim) - 1]],
          drop (nDim + 1) slices
        ]
  )
  where
    nDim = normalizeItem (V.length shape) dim
    newShape = V.concat [
        V.take nDim shape,
        singleton 1,
        V.drop (nDim + 1) shape
      ]

insertDim :: (Unbox t) => Tensor t -> Int -> Tensor t
insertDim (Tensor shape dat) dim =
  Tensor newShape dat
  where
    nDim = normalizeItem (V.length shape) dim
    newShape = V.concat [
        V.take nDim shape,
        singleton 1,
        V.drop nDim shape
      ]

-- | tensor -> dim -> times
repeatAlongDim :: (Unbox t) => Tensor t -> Int -> Int -> Tensor t
repeatAlongDim x@(Tensor shape _) dim times =
  tensor newShape (
    \ index ->
        x !? (index // [(nDim, (index ! nDim) `mod` currentDim)])
  )
  where
    nDim = normalizeItem (V.length shape) dim
    currentDim = shape ! nDim
    newShape = shape // [(nDim, currentDim * times)]
    -- debugPrint x =
    --   unsafePerformIO $ print x >> return x

-- | tensor -> dims -> times
repeatAlongDims :: (Unbox t) =>
  Tensor t -> [Int] -> [Int] -> Tensor t
repeatAlongDims x dims timess =
  foldl' (
    \ accum (dim, times) ->
      repeatAlongDim accum dim times
  ) x $ zip dims timess

swapDim :: (Unbox t) =>
  Tensor t -> Int -> Int -> Tensor t
swapDim x@(Tensor shape _) from to =
  tensor (swapElementsAt nFrom nTo shape) ((x !?) . swapElementsAt nFrom nTo)
  where
    nFrom = normalizeItem (V.length shape) from
    nTo = normalizeItem (V.length shape) to

{-# INLINE tensor #-}
{-# INLINE full #-}
{-# INLINE zeros #-}
{-# INLINE ones #-}
{-# INLINE single #-}
{-# INLINE eye #-}
{-# INLINE randn #-}
{-# INLINE arange #-}

{-# INLINE map #-}
{-# INLINE elementwise #-}
{-# INLINE performWithBroadcasting #-}
{-# INLINE verifyBroadcastable #-}
{-# INLINE broadcast #-}

{-# INLINE getElem #-}
{-# INLINE slice #-}
{-# INLINE advancedIndex #-}

{-# INLINE numel #-}
{-# INLINE item #-}
{-# INLINE flatten #-}
{-# INLINE sum #-}
{-# INLINE sumBabushka #-}
{-# INLINE mean #-}
{-# INLINE sumAlongDim #-}
{-# INLINE sumAlongDimKeepDims #-}
{-# INLINE insertDim #-}
{-# INLINE repeatAlongDim #-}
{-# INLINE repeatAlongDims #-}
{-# INLINE swapDim #-}
