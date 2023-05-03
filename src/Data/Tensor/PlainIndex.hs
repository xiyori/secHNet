module Data.Tensor.PlainIndex where

import Data.Vector.Storable (Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.Tensor.Definitions

import Foreign
import Foreign.C.Types


-- | Coefficients for index flattening.
--
--   Signature: @sizeOfElem -> shape -> stride@
computeStride :: CSize -> Shape -> Index
computeStride sizeOfElem shape =
  case V.length shape of {len ->
    V.constructrN len (
      \ accum ->
        if V.null accum then
          fromIntegral sizeOfElem
        else fromIntegral (shape ! (len - V.length accum)) * V.head accum
    )
  }

-- | Parse negative index value.
--
--   Signature: @dim -> i -> normI@
normalizeItem :: (Integral a, Integral b) => a -> b -> b
normalizeItem dim i =
  if i < 0 then
    if dim == 0 && i == -1 then
      0
    else fromIntegral dim + i
  else i

-- | Parse negative index values.
--
--   Signature: @shape -> index -> normIndex@
normalizeIndex :: Shape -> Index -> Index
normalizeIndex = V.zipWith normalizeItem

-- | Total number of elements in a tensor with shape @shape@.
--
--   Signature: @shape -> numel@
totalElems :: Shape -> CSize
totalElems = V.foldl' (*) 1

-- | Validate index correctness.
--
--   Signature: @shape -> index -> isValid@
validateIndex :: Shape -> Index -> Bool
validateIndex shape index
  | V.length shape /= V.length index  = False
  | V.any (< 0) index                 = False
  | V.or $ V.zipWith ((>=) . fromIntegral) index shape = False
  | otherwise                         = True

-- | Determine if two shapes can be broadcasted.
--
--   Signature: @shape1 -> shape2 -> canBroadcast@
verifyBroadcastable :: Shape -> Shape -> Bool
verifyBroadcastable shape1 shape2 =
  V.and (
    V.zipWith (
      \ dim1 dim2 -> dim1 == dim2 || dim1 == 1 || dim2 == 1
    ) (V.reverse shape1) (V.reverse shape2)
  )

-- | Broadcast shapes and strides.
--
--   Signature: @shape1 -> shape2 -> stride1 -> stride2 ->
--                (shape1, shape2, stride1, stride2)@
broadcastShapesStrides ::
  Shape -> Shape -> Index -> Index -> (Shape, Shape, Index, Index)
broadcastShapesStrides shape1 shape2 stride1 stride2 =
  case V.length shape1 of {nDims1 ->
  case V.length shape2 of {nDims2 ->
    (V.concat [V.take (nDims2 - nDims1) shape2, shape1],
     V.concat [V.take (nDims1 - nDims2) shape1, shape2],
     V.concat [V.replicate (nDims2 - nDims1) 0, stride1],
     V.concat [V.replicate (nDims1 - nDims2) 0, stride2])
  }}

-- | Resulting shape after broadcasting.
--
--   Signature: @shape1 -> shape2 -> newShape@
broadcastedShape :: Shape -> Shape -> Shape
broadcastedShape =
  V.zipWith (
      \ dim1 dim2 ->
      if dim1 == 1 then
        dim2
      else dim1
  )

-- | Determine if all elements in a list are equal.
allEqual :: Eq a => [a] -> Bool
allEqual xs = all (== head xs) $ tail xs

-- | Swap index dimensions.
--
--   Signature: @index -> dim1 -> dim2 -> swappedIndex@
swapElementsAt :: (Storable t) => Vector t -> Int -> Int -> Vector t
swapElementsAt index dim1 dim2 =
  index // [(dim1, index ! dim2), (dim2, index ! dim1)]

{-# INLINE computeStride #-}
{-# INLINE normalizeItem #-}
{-# INLINE normalizeIndex #-}
{-# INLINE totalElems #-}
{-# INLINE validateIndex #-}
{-# INLINE verifyBroadcastable #-}
{-# INLINE broadcastShapesStrides #-}
{-# INLINE broadcastedShape #-}
{-# INLINE allEqual #-}
{-# INLINE swapElementsAt #-}
