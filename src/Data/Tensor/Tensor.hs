{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Data.Tensor.Tensor where

import Control.Applicative
import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List (foldl')
import System.Random
import Data.Random.Normal
import Data.Tensor.Index
import Data.Tensor.Size

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx <> C.funCtx)
-- Include C utils.
C.include "cbits/cbits.h"

-- | Tensor data type.
data (Storable t) =>
  Tensor t = Tensor {
    -- | Tensor shape.
    shape :: !Index,
    -- | Tensor stride in bytes, analogous to NumPy array stride.
    stride :: !Index,
    -- | Data offset in bytes.
    offset :: !CInt,
    -- | Internal data representation.
    tensorData :: !(Vector t)
  } deriving Show

-- | Slice data type.
data Slice
  = Int -- | Slice @start :. end@.
        :. Int
  | Slice -- | Slice @start :. end :| step@.
          :| Int
  -- | Single index @I index@.
  | I Int
  -- | Full slice, analogous to NumPy @:@.
  | A
  -- | Insert new dim, analogous to NumPy @None@.
  | None
  -- | Ellipses, analogous to NumPy @...@.
  | Ell

-- | Slice indexer data type.
type Slices = [Slice]

-- | Advanced indexer data type.
type TensorIndex = [Tensor CInt]


-- instance Num (Tensor CFloat) where
--   (+) = performWithBroadcasting (+)
--   (-) = performWithBroadcasting (-)
--   (*) = performWithBroadcasting (*)
--   abs = Data.Tensor.Tensor.map abs
--   signum = Data.Tensor.Tensor.map signum
--   fromInteger = single . fromInteger
--   {-# INLINE (+) #-}
--   {-# INLINE (-) #-}
--   {-# INLINE (*) #-}
--   {-# INLINE abs #-}
--   {-# INLINE signum #-}
--   {-# INLINE fromInteger #-}

-- instance (Storable t, Fractional t) => Fractional (Tensor t) where
--   (/) = performWithBroadcasting (/)
--   fromRational = single . fromRational
--   {-# INLINE (/) #-}
--   {-# INLINE fromRational #-}

-- instance (Storable t, Floating t) => Floating (Tensor t) where
--   pi = single pi
--   exp = Data.Tensor.Tensor.map exp
--   log = Data.Tensor.Tensor.map log
--   sin = Data.Tensor.Tensor.map sin
--   cos = Data.Tensor.Tensor.map cos
--   asin = Data.Tensor.Tensor.map asin
--   acos = Data.Tensor.Tensor.map acos
--   atan = Data.Tensor.Tensor.map atan
--   sinh = Data.Tensor.Tensor.map sinh
--   cosh = Data.Tensor.Tensor.map cosh
--   asinh = Data.Tensor.Tensor.map asinh
--   acosh = Data.Tensor.Tensor.map acosh
--   atanh = Data.Tensor.Tensor.map atanh
--   {-# INLINE pi #-}
--   {-# INLINE exp #-}
--   {-# INLINE log #-}
--   {-# INLINE sin #-}
--   {-# INLINE cos #-}
--   {-# INLINE asin #-}
--   {-# INLINE acos #-}
--   {-# INLINE atan #-}
--   {-# INLINE sinh #-}
--   {-# INLINE cosh #-}
--   {-# INLINE asinh #-}
--   {-# INLINE acosh #-}
--   {-# INLINE atanh #-}

-- | Generate a tensor from a generator function.
--
--   Signature: @shape -> generator -> tensor@
tensor :: Storable t => Index -> (Int -> t) -> Tensor t
tensor shape builder =
  case V.generate (fromIntegral $ totalElems shape) builder of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }


-- | Return a new tensor filled with @fillValue@.
--
--   Signature: @shape -> fillValue -> tensor@
full :: Storable t => Index -> t -> Tensor t
full shape fillValue =
  case V.replicate (fromIntegral $ totalElems shape) fillValue of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }

-- | Return a new tensor filled with zeros.
--
--   Signature: @shape -> tensor@
zeros :: (Storable t, Num t) => Index -> Tensor t
zeros shape = full shape 0


-- | Return a new tensor filled with ones.
--
--   Signature: @shape -> tensor@
ones :: (Storable t, Num t) => Index -> Tensor t
ones shape = full shape 1


-- | Return a new tensor of shape [1] with a single value.
--
--   Signature: @value -> tensor@
single :: (Storable t, Num t) => t -> Tensor t
single = full $ V.singleton 1

-- | Return a 2-D float tensor with ones
--   on the diagonal and zeros elsewhere.
--
--   @diagonalIndex@ 0 refers to the main diagonal,
--   a positive value refers to an upper diagonal,
--   and a negative value to a lower diagonal.
--
--   Signature: @rows -> columns -> diagonalIndex@
eyeF :: CInt -> CInt -> CInt -> Tensor CFloat
eyeF rows columns diagonalIndex =
  Tensor (V.fromList [rows, columns])
  (V.fromList [columns * sizeOfCFloat, sizeOfCFloat]) 0
  $ unsafePerformIO
  $ do
    mutableData <- VM.new $ fromIntegral $ rows * columns
    VM.unsafeWith mutableData (
      \ mutableDataPtr ->
        [CU.exp| void {
          eye_f(
            $(int rows),
            $(int columns),
            $(int diagonalIndex),
            $(float *mutableDataPtr)
          )
        } |]
      )
    V.unsafeFreeze mutableData

-- randn :: (Storable t, RandomGen g, Random t, Floating t) =>
--   Index -> g -> (Tensor t, g)
-- randn shape gen =
--   (Tensor shape randomVector, newGen)
--   where
--     n = countIndex shape
--     (randomList, newGen) = go n [] gen
--     go 0 xs g = (xs, g)
--     go count xs g =
--       let (randomValue, newG) = normal g in
--         go (count - 1) (randomValue : xs) newG
--     randomVector = fromList randomList

-- | Return evenly spaced float values within a given interval.
--
--   Example: @arange 0 3 1 = tensor([0, 1, 2])@
--
--   Signature: @low -> high -> step -> tensor@
arangeF :: CFloat -> CFloat -> CFloat -> Tensor CFloat
arangeF low high step =
  tensor (V.singleton $ floor $ (high - low) / step) (
    \ fIndex ->
      low + step * fromIntegral fIndex
  )


-- elementwise :: (Storable a, Storable b, Storable c) =>
--   (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
-- elementwise f (Tensor shape dat1) (Tensor _ dat2) =
--   Tensor shape $ V.zipWith f dat1 dat2

-- performWithBroadcasting :: (Storable a, Storable b, Storable c) =>
--   (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
-- performWithBroadcasting f x1@(Tensor shape1 _) x2@(Tensor shape2 _)
--   | V.head shape1 == 1 && V.length shape1 == 1 =
--     Data.Tensor.map (f $ item x1) x2
--   | V.head shape2 == 1 && V.length shape2 == 1 =
--     Data.Tensor.map (flip f $ item x2) x1
--   | otherwise =
--     uncurry (elementwise f) $ broadcast x1 x2

-- verifyBroadcastable :: (Storable a, Storable b) =>
--   Tensor a -> Tensor b -> Bool
-- verifyBroadcastable (Tensor shape1 _) (Tensor shape2 _) =
--   V.length shape1 == V.length shape2
--   && V.and (
--     V.zipWith (
--       \ dim1 dim2 -> dim1 == dim2 || dim1 == 1 || dim2 == 1
--     ) shape1 shape2
--   )

-- broadcast :: (Storable a, Storable b) =>
--   Tensor a -> Tensor b -> (Tensor a, Tensor b)
-- broadcast x1@(Tensor shape1 _) x2@(Tensor shape2 _) =
  -- | verifyBroadcastable x1 x2 =
  --   (repeatAlongDims x1 dims1 times1,
  --    repeatAlongDims x2 dims2 times2)
  -- | otherwise =
  --   error
  --   $ "tensors "
  --   ++ show shape1
  --   ++ " "
  --   ++ show shape2
  --   ++ " can not be broadcasted"
  -- where
  --   zipped = V.zip3 (fromList [0 .. V.length shape1 - 1]) shape1 shape2
  --   (dims1, times1) = V.foldr addDim ([], []) zipped
  --   (dims2, times2) = V.foldr (addDim . flipDims) ([], []) zipped
  --   addDim (i, dim1, dim2) accum@(accumDims, accumTimes) =
  --     if dim1 /= dim2 && dim1 == 1 then
  --       (i : accumDims, dim2 : accumTimes)
  --     else accum
  --   flipDims (i, dim1, dim2) = (i, dim2, dim1)

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

-- | Get element of a tensor without cheking for index validity.
--
--   This function is not safe to use,
--   consider using @(!)@ operator instead.
--
--   Signature: @tensor -> index -> elem@
unsafeGetElem :: Storable t => Tensor t -> Index -> t
unsafeGetElem (Tensor _ stride offset dat) index =
  case V.unsafeCast dat of {dataCChar ->
    unsafeDupablePerformIO
    $ [CU.exp| char * {
        get_elem(
          $vec-len:stride,
          $vec-ptr:(int *stride),
          $(int offset),
          $vec-ptr:(char *dataCChar),
          $vec-ptr:(int *index)
        )
      } |] >>= peek . castPtr
  }

-- | Get element of a tensor.
--
--   Signature: @tensor -> index -> elem@
(!) :: Storable t => Tensor t -> [CInt] -> t
(!) x@(Tensor shape _ _ _) index =
  case normalizeIndex shape $ V.fromList index of {nIndex ->
    if validateIndex shape nIndex then
      unsafeGetElem x nIndex
    else
      error
      $ "incorrect index "
      ++ show index
      ++ " for shape "
      ++ show shape
  }

-- slice :: (Storable t) => Tensor t -> Slices -> Tensor t
-- slice x@(Tensor shape _) slices =
--   tensor newShape (
--     \ index ->
--       x !? fromList (zipWith (!) expandedIndex $ toList index)
--   )
--   where
--     expandedIndex =
--       zipWith (
--         \ dim slice ->
--           if V.null slice then
--             fromList [0 .. dim - 1]
--           else slice
--       ) (toList shape) slices
--     newShape = fromList $ Prelude.map V.length expandedIndex

-- (!:) :: (Storable t) => Tensor t -> Slices -> Tensor t
-- (!:) = slice

-- validateTensorIndex :: TensorIndex -> Bool
-- validateTensorIndex = allEqual . Prelude.map shape

-- advancedIndex :: (Storable t) => Tensor t -> TensorIndex -> Tensor t
-- advancedIndex x tensorIndex
--   | validateTensorIndex tensorIndex =
--     tensor (shape $ head tensorIndex) (
--       \ index ->
--         x !? fromList (Prelude.map (!? index) tensorIndex)
--     )
--   | otherwise =
--     error
--     $ "incorrect index "
--     ++ show tensorIndex
--     ++ " for shape "
--     ++ show (shape x)

-- (!.) :: (Storable t) => Tensor t -> TensorIndex -> Tensor t
-- (!.) = advancedIndex

-- | Return the element of a @single@ tensor.
--
--   Signature: @tensor -> elem@
item :: (Storable t) => Tensor t -> t
item x@(Tensor shape _ _ _)
  | V.length shape == 1 && V.head shape == 1 =
    unsafeGetElem x $ V.singleton 0
  | otherwise =
    error
    $ "cannot get item from tensor with shape "
    ++ show shape


-- | Total number of elements in a tensor.
--
--   Signature: @tensor -> numel@
numel :: (Storable t) => Tensor t -> CInt
numel (Tensor shape _ _ _) = totalElems shape

-- | Return a contiguous copy of a tensor.
--
--   Signature: @tensor -> copiedTensor@
copy :: (Storable t) => Tensor t -> Tensor t
copy (Tensor shape stride offset dat) =
  case sizeOfElem dat of {elemSize ->
  case V.unsafeCast dat of {dataCChar ->
    -- If tensor is not contiguous, deep copy
    if computeStride elemSize shape /= stride then
      Tensor shape
      (computeStride elemSize shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            copy(
              $vec-len:stride,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride),
              $(int offset),
              $(int elemSize),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    -- If tensor is already contiguous, take a view
    else
      Tensor shape stride 0
        $ V.unsafeCast
        $ V.slice (fromIntegral offset)
          (fromIntegral $ totalElems shape * elemSize) dataCChar
  }}

-- | Return a tensor with last 2 axes transposed.
--
--   Signature: @tensor -> tensor@
transpose :: Storable t => Tensor t -> Tensor t
transpose x = swapDims x (-1) (-2)

-- | Return a copy of the tensor collapsed into one dimension.
--
--   Signature: @tensor -> tensor@
flatten :: (Storable t) => Tensor t -> Tensor t
flatten x =
  case copy x of {(Tensor shape stride offset dat) ->
    Tensor (V.singleton $ totalElems shape)
      (V.singleton $ V.last stride) offset dat
  }

-- | Map a function over a tensor.
--
--   Signature: @function -> tensor -> tensor@
map :: (Storable a, Storable b) => (a -> b) -> Tensor a -> Tensor b
map f x =
  case copy x of {(Tensor shape stride offset dat) ->
    Tensor shape stride offset
    $ V.map f dat
  }

-- | Left fold with strict accumulator.
--
--   Signature: @function -> accum -> tensor -> result@
foldl' :: Storable b => (a -> b -> a) -> a -> Tensor b -> a
foldl' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldl' f accum dat
  }

-- | Right fold with strict accumulator.
--
--   Signature: @function -> accum -> tensor -> result@
foldr' :: Storable b => (b -> a -> a) -> a -> Tensor b -> a
foldr' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldr' f accum dat
  }

-- | Sum elements of a tensor.
--
--   Signature: @function -> tensor -> sum@
sum :: (Storable t, Num t) => Tensor t -> t
sum x =
  case copy x of {(Tensor shape stride offset dat) ->
    fst $ V.foldl' kahanSum (0, 0) dat
      where
        kahanSum (sum, c) item =
          case item - c of {y ->
          case sum + y of {t ->
            (t, (t - sum) - y)
          }}
  }

-- | Sum elements of a float tensor more efficiently.
--
--   Signature: @function -> tensor -> sum@
sumF :: Tensor CFloat -> CFloat
sumF (Tensor shape stride offset dat) =
  case V.unsafeCast dat of {dataCChar ->
    [CU.pure| float {
      sum_f(
        $vec-len:stride,
        $vec-ptr:(int *shape),
        $vec-ptr:(int *stride),
        $(int offset),
        $vec-ptr:(char *dataCChar)
      )
    } |]
  }

-- sumBabushka :: (Storable t, Num t, Ord t) => Tensor t -> t
-- sumBabushka (Tensor _ dat) =
--   fst $ V.foldl' kahanBabushkaSum (0, 0) dat
--   where
--     kahanBabushkaSum (sum, c) item =
--       if abs sum >= abs item then
--         let y = item - c
--             t = sum + y in
--           (t, (t - sum) - y)
--       else
--         let y = sum - c
--             t = item + y in
--           (t, (t - item) - y)

mean :: (Storable t, Fractional t) => Tensor t -> t
mean x = Data.Tensor.Tensor.sum x / fromIntegral (numel x)

-- sumAlongDim :: (Storable t, Num t) =>
--   Tensor t -> Int -> Tensor t
-- sumAlongDim x@(Tensor shape _) dim =
--   tensor newShape (
--     \ index ->
--       let slices = Prelude.map singleton $ toList index in
--         Data.Tensor.sum . slice x
--         $ concat [
--           take nDim slices,
--           [fromList [0 .. (shape ! nDim) - 1]],
--           drop nDim slices
--         ]
--   )
--   where
--     nDim = normalizeItem (V.length shape) dim
--     newShape = V.concat [
--         V.take nDim shape,
--         V.drop (nDim + 1) shape
--       ]

-- sumAlongDimKeepDims :: (Storable t, Num t) =>
--   Tensor t -> Int -> Tensor t
-- sumAlongDimKeepDims x@(Tensor shape _) dim =
--   tensor newShape (
--     \ index ->
--       let slices = Prelude.map singleton $ toList index in
--         Data.Tensor.sum . slice x
--         $ concat [
--           take nDim slices,
--           [fromList [0 .. (shape ! nDim) - 1]],
--           drop (nDim + 1) slices
--         ]
--   )
--   where
--     nDim = normalizeItem (V.length shape) dim
--     newShape = V.concat [
--         V.take nDim shape,
--         singleton 1,
--         V.drop (nDim + 1) shape
--       ]

-- insertDim :: (Storable t) => Tensor t -> Int -> Tensor t
-- insertDim (Tensor shape dat) dim =
--   Tensor newShape dat
--   where
--     nDim = normalizeItem (V.length shape) dim
--     newShape = V.concat [
--         V.take nDim shape,
--         singleton 1,
--         V.drop nDim shape
--       ]

-- -- | tensor -> dim -> times
-- repeatAlongDim :: (Storable t) => Tensor t -> Int -> Int -> Tensor t
-- repeatAlongDim x@(Tensor shape _) dim times =
--   tensor newShape (
--     \ index ->
--         x !? (index // [(nDim, (index ! nDim) `mod` currentDim)])
--   )
--   where
--     nDim = normalizeItem (V.length shape) dim
--     currentDim = shape ! nDim
--     newShape = shape // [(nDim, currentDim * times)]
--     -- debugPrint x =
--     --   unsafePerformIO $ print x >> return x

-- -- | tensor -> dims -> times
-- repeatAlongDims :: (Storable t) =>
--   Tensor t -> [Int] -> [Int] -> Tensor t
-- repeatAlongDims x dims timess =
--   foldl' (
--     \ accum (dim, times) ->
--       repeatAlongDim accum dim times
--   ) x $ zip dims timess

-- | Interchange two dims of a tensor.
--
--   Signature: @tensor -> i -> j -> swappedTensor@
swapDims :: (Storable t) =>
  Tensor t -> Int -> Int -> Tensor t
swapDims x@(Tensor shape stride offset dat) i j =
  case V.length shape of {nDims ->
  case normalizeItem nDims i of {nI ->
  case normalizeItem nDims j of {nJ ->
    if 0 <= nI && nI < nDims && 0 <= nJ && nJ < nDims then
      Tensor (swapElementsAt shape nI nJ) (swapElementsAt stride nI nJ)
      offset dat
    else
      error
      $ "cannot swap dims "
      ++ show i
      ++ ", "
      ++ show j
      ++ " of tensor with shape "
      ++ show shape
  }}}

{-# INLINE tensor #-}
{-# INLINE full #-}
-- {-# INLINE zeros #-}
-- {-# INLINE ones #-}
-- {-# INLINE single #-}
{-# INLINE eyeF #-}
-- {-# INLINE randn #-}
-- {-# INLINE arangeF #-}

-- {-# INLINE elementwise #-}
-- {-# INLINE performWithBroadcasting #-}
-- {-# INLINE verifyBroadcastable #-}
-- {-# INLINE broadcast #-}

{-# INLINE unsafeGetElem #-}
-- {-# INLINE slice #-}
-- {-# INLINE advancedIndex #-}
-- {-# INLINE item #-}

-- {-# INLINE numel #-}
{-# INLINE copy #-}
-- {-# INLINE transpose #-}
-- {-# INLINE flatten #-}
{-# INLINE map #-}
{-# INLINE foldl' #-}
{-# INLINE foldr' #-}
{-# INLINE sum #-}
{-# INLINE sumF #-}
-- {-# INLINE sumBabushka #-}
-- {-# INLINE mean #-}
-- {-# INLINE sumAlongDim #-}
-- {-# INLINE sumAlongDimKeepDims #-}
-- {-# INLINE insertDim #-}
-- {-# INLINE repeatAlongDim #-}
-- {-# INLINE repeatAlongDims #-}
-- {-# INLINE swapDims #-}
