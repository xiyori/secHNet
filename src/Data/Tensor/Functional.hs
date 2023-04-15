{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Data.Tensor.Functional where

import Control.Monad.IO.Class
import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List (foldl')
import System.Random
import Data.Random.Normal
import Data.Tensor.Index
import Data.Tensor.Size
import Data.Tensor.Definitions as T

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx)
-- Include C utils.
C.include "cbits/cbits.h"


-- Construction
-- ------------

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

-- | Return a new tensor of shape [] with a single value.
scalar :: (Storable t, Num t) => t -> Tensor t
scalar = full V.empty

-- | Return a new tensor of shape [1] with a single value.
single :: (Storable t, Num t) => t -> Tensor t
single = full $ V.singleton 1

-- | Return a new tensor filled with random values from
--   standard normal distribution.
--
--   Signature: @shape -> gen -> (tensor, gen)@
randn :: (Storable t, Random t, Floating t, RandomGen g) =>
  Index -> g -> (Tensor t, g)
randn shape gen =
  case V.unfoldrExactN (fromIntegral $ totalElems shape)
  normal gen of {dat ->
    (Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat,
     gen)
  }

-- | Return a new tensor filled with random values from
--   standard normal distribution inside IO monad.
--
--   Signature: @shape -> tensor@
randnM :: (Storable t, Random t, Floating t, MonadIO m) =>
  Index -> m (Tensor t)
randnM shape = do
  dat <- V.replicateM (fromIntegral $ totalElems shape) (
    getStdGen >>= (
      \ gen ->
        case normal gen of {(value, gen) ->
          setStdGen gen >> return value
        }
    ))
  return $ Tensor shape
    (computeStride (sizeOfElem dat) shape) 0 dat

-- | Generate a tensor from a generator function.
--
--   Signature: @tensor -> generator -> tensor@
tensorLike :: (Storable a, Storable b) => Tensor a -> (Int -> b) -> Tensor b
tensorLike (Tensor shape _ _ _) = tensor shape

-- | Return a new tensor filled with @fillValue@.
--
--   Signature: @tensor -> fillValue -> tensor@
fullLike :: (Storable a, Storable b) => Tensor a -> b -> Tensor b
fullLike (Tensor shape _ _ _) = full shape

-- | Return a new tensor filled with zeros.
--
--   Signature: @tensor -> tensor@
zerosLike :: (Storable a, Storable b, Num b) => Tensor a -> Tensor b
zerosLike (Tensor shape _ _ _) = full shape 0

-- | Return a new tensor filled with ones.
--
--   Signature: @tensor -> tensor@
onesLike :: (Storable a, Storable b, Num b) => Tensor a -> Tensor b
onesLike (Tensor shape _ _ _) = full shape 1


-- Binary operations
-- -----------------

-- | Broadcast tensors without copying.
broadcast :: (Storable a, Storable b) =>
  Tensor a -> Tensor b -> (Tensor a, Tensor b)
broadcast (Tensor shape1 stride1 offset1 dat1)
          (Tensor shape2 stride2 offset2 dat2)
  | verifyBroadcastable shape1 shape2 =
    case V.length shape1 of {nDims1 ->
    case V.length shape2 of {nDims2 ->
    case (
      V.concat [V.take (nDims2 - nDims1) shape2, shape1],
      V.concat [V.take (nDims1 - nDims2) shape1, shape2],
      V.concat [V.replicate (nDims2 - nDims1) 0, stride1],
      V.concat [V.replicate (nDims1 - nDims2) 0, stride2]
    ) of {(shape1, shape2, stride1, stride2) ->
    case V.zipWith (
      \ dim1 dim2 ->
        if dim1 == 1 then
          dim2
        else dim1
    ) shape1 shape2 of {newShape ->
      (
        Tensor newShape
        (V.zipWith (
          \ dim stride ->
            if dim == 1 then
              0
            else stride
        ) shape1 stride1) offset1 dat1,
        Tensor newShape
        (V.zipWith (
          \ dim stride ->
            if dim == 1 then
              0
            else stride
        ) shape2 stride2) offset2 dat2
      )
    }}}}
  | otherwise =
    error
    $ "tensors of shapes "
    ++ show shape1
    ++ ", "
    ++ show shape2
    ++ " can not be broadcasted"

-- | Perform elementwise operation without cheking
--   for validity of arguments.
--
--   This function is not safe to use,
--   consider using @elementwise@ instead.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
unsafeElementwise :: (Storable a, Storable b, Storable c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
unsafeElementwise f x1 x2 =
  case copy x1 of {(Tensor shape stride offset dat1) ->
  case copy x2 of {(Tensor _ _ _ dat2) ->
    Tensor shape stride offset
    $ V.zipWith f dat1 dat2
  }}

-- | Perform elementwise operation with broadcasting.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
elementwise :: (Storable a, Storable b, Storable c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
elementwise f x1 x2 =
  case broadcast x1 x2 of {(x1, x2) ->
    if totalElems (shape x1) == 1 then
      Data.Tensor.Functional.map (f $ item x1) x2
    else if totalElems (shape x2) == 1 then
      Data.Tensor.Functional.map (flip f $ item x2) x1
    else unsafeElementwise f x1 x2
  }

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

-- | True if two tensors have the same shape and elements, False otherwise.
tensorEqual :: (Storable t) => Tensor t -> Tensor t -> Bool
tensorEqual (Tensor shape stride1 offset1 dat1)
            (Tensor shape2 stride2 offset2 dat2)
  | shape == shape2 =
    case sizeOfElem dat1 of {elemSize ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      toBool [CU.pure| int {
        equal(
          $vec-len:shape,
          $vec-ptr:(int *shape),
          $(int elemSize),
          $vec-ptr:(int *stride1),
          $(int offset1),
          $vec-ptr:(char *data1CChar),
          $vec-ptr:(int *stride2),
          $(int offset2),
          $vec-ptr:(char *data2CChar)
        )
      } |]
    }}}
  | otherwise = False

-- Return (x1 == x2) element-wise.
-- equal :: (Storable t) => Tensor t -> Tensor t -> Tensor CBool
-- equal


-- Indexing
-- --------

-- | Get element of a tensor without cheking for index validity.
--
--   This function is not safe to use,
--   consider using @(!)@ operator instead.
unsafeGetElem :: Storable t => Tensor t -> Index -> t
unsafeGetElem (Tensor shape stride offset dat) index =
  case V.unsafeCast dat of {dataCChar ->
    unsafeDupablePerformIO
    $ [CU.exp| char * {
        get_elem(
          $vec-len:shape,
          $vec-ptr:(int *stride),
          $(int offset),
          $vec-ptr:(char *dataCChar),
          $vec-ptr:(int *index)
        )
      } |] >>= peek . castPtr
  }

-- | Get element of a tensor.
(!) :: Storable t => Tensor t -> [CInt] -> t
(!) x@(Tensor shape _ _ _) index =
  case normalizeIndex shape $ V.fromList index of {normIndex ->
    if validateIndex shape normIndex then
      unsafeGetElem x normIndex
    else
      error
      $ "incorrect index "
      ++ show index
      ++ " for shape "
      ++ show shape
  }

-- slice :: (Storable t) => Tensor t -> Slices -> Tensor t
-- slice x@(Tensor shape stride offset dat) slices =
--   case To

-- | An infix synonym for slice.
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

-- | Return the value of a tensor with one element.
item :: (Storable t) => Tensor t -> t
item x@(Tensor shape _ _ _)
  | totalElems shape == 1 =
      unsafeGetElem x $ V.singleton 0
  | otherwise =
    error
    $ "cannot get item from tensor of shape "
    ++ show shape


-- Unary operations
-- ----------------

-- | Total number of elements in a tensor.
numel :: (Storable t) => Tensor t -> CInt
numel (Tensor shape _ _ _) = totalElems shape

-- | Take a mean of elements in a tensor.
mean :: (Storable t, FractionalTensor t) => Tensor t -> t
mean x = T.sum x / fromIntegral (numel x)

-- | Return a contiguous copy of a tensor.
copy :: (Storable t) => Tensor t -> Tensor t
copy (Tensor shape stride offset dat) =
  case sizeOfElem dat of {elemSize ->
  case V.unsafeCast dat of {dataCChar ->
  case computeStride elemSize shape of {contiguousStride ->
    -- If tensor is not contiguous, deep copy
    if contiguousStride /= stride then
      Tensor shape contiguousStride 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            copy(
              $vec-len:shape,
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
  }}}

-- | Return a tensor with last 2 axes transposed.
transpose :: Storable t => Tensor t -> Tensor t
transpose x = swapDims x (-1) (-2)

-- | Return a copy of the tensor collapsed into one dimension.
flatten :: (Storable t) => Tensor t -> Tensor t
flatten x =
  case copy x of {(Tensor shape stride offset dat) ->
    Tensor (V.singleton $ totalElems shape)
      (V.singleton $ sizeOfElem dat) offset dat
  }

-- | Give a new shape to a tensor without changing its data.
--
--   Signature: @tensor -> newShape -> tensor@
view :: (Storable t) => Tensor t -> Index -> Tensor t
view x@(Tensor shape stride offset dat) newShape
  | totalElems shape == totalElems newShape =
    case sizeOfElem dat of {elemSize ->
    case computeStride elemSize shape of {contiguousStride ->
    case computeStride elemSize newShape of {newStride ->
      if stride == contiguousStride then
        Tensor newShape newStride offset dat
      else
        case copy x of {(Tensor _ _ offset dat) ->
          Tensor newShape newStride offset dat
        }
    }}}
  | otherwise =
    error
    $ "cannot reshape tensor of shape "
    ++ show shape
    ++ " into shape "
    ++ show newShape

-- | Map a function over a tensor.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
map :: (Storable a, Storable b) => (a -> b) -> Tensor a -> Tensor b
map f x =
  case copy x of {(Tensor shape stride offset dat) ->
    Tensor shape stride offset
    $ V.map f dat
  }

-- | Left fold with strict accumulator.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
foldl' :: Storable b => (a -> b -> a) -> a -> Tensor b -> a
foldl' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldl' f accum dat
  }

-- | Right fold with strict accumulator.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
foldr' :: Storable b => (b -> a -> a) -> a -> Tensor b -> a
foldr' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldr' f accum dat
  }

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

-- | Insert a new dim into a tensor.
--
--   Signature: @tensor -> dim -> tensor@
insertDim :: (Storable t) => Tensor t -> Int -> Tensor t
insertDim (Tensor shape stride offset dat) dim =
  case V.length shape of {nDims ->
  case normalizeItem nDims dim of {normDim ->
    if 0 <= normDim && normDim <= nDims then
      Tensor
      (V.concat [
          V.take normDim shape,
          V.singleton 1,
          V.drop normDim shape])
      (V.concat [
          V.take normDim stride,
          V.singleton 0,
          V.drop normDim stride])
      offset dat
    else
      error
      $ "cannot insert dim "
      ++ show dim
      ++ " in tensor of shape "
      ++ show shape
  }}

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

-- | Interchange two dims of a tensor.
--
--   Signature: @tensor -> dim1 -> dim2 -> tensor@
swapDims :: (Storable t) =>
  Tensor t -> Int -> Int -> Tensor t
swapDims x@(Tensor shape stride offset dat) dim1 dim2 =
  case V.length shape of {nDims ->
  case normalizeItem nDims dim1 of {normDim1 ->
  case normalizeItem nDims dim2 of {normDim2 ->
    if 0 <= normDim1 && normDim1 < nDims &&
       0 <= normDim2 && normDim2 < nDims then
      Tensor
      (swapElementsAt shape normDim1 normDim2)
      (swapElementsAt stride normDim1 normDim2)
      offset dat
    else
      error
      $ "cannot swap dims "
      ++ show dim1
      ++ ", "
      ++ show dim2
      ++ " of tensor of shape "
      ++ show shape
  }}}

{-# INLINE tensor #-}
{-# INLINE full #-}
-- {-# INLINE zeros #-}
-- {-# INLINE ones #-}
-- {-# INLINE scalar #-}
-- {-# INLINE single #-}
{-# INLINE randn #-}
{-# INLINE randnM #-}

{-# INLINE unsafeElementwise #-}
-- {-# INLINE elementwise #-}
{-# INLINE broadcast #-}
-- {-# INLINE dot #-}
{-# INLINE tensorEqual #-}

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
-- {-# INLINE mean #-}
-- {-# INLINE sumAlongDim #-}
-- {-# INLINE sumAlongDimKeepDims #-}
-- {-# INLINE insertDim #-}
-- {-# INLINE repeatAlongDim #-}
-- {-# INLINE swapDims #-}
