module Data.Tensor.PlainIndex where

import Data.Vector.Storable (Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.List
import Data.Tensor.Definitions

import Foreign
import Foreign.C.Types


-- | Convert an internal shape to a list representation.
--
--   Signature: @shape -> listShape@
shapeToIndex :: Shape -> Index
shapeToIndex = V.toList . V.map fromIntegral

-- | Convert a list representation to an internal shape.
--
--   Signature: @listShape -> shape@
indexToShape :: Index -> Shape
indexToShape listShape
  | all (>= 0) listShape =
    V.map fromIntegral $ V.fromList listShape
  | otherwise =
    error
    $ "invalid tensor shape "
    ++ show listShape

-- | Total number of elements in a tensor with shape @shape@.
--
--   Signature: @shape -> numel@
totalElems :: Index -> Int
totalElems = foldl' (*) 1

-- | Total number of elements in a tensor with shape @shape@.
--
--   Signature: @shape -> numel@
totalElems_ :: Shape -> CSize
totalElems_ = V.foldl' (*) 1

-- | Coefficients for index flattening.
--
--   Signature: @sizeOfElem -> shape -> stride@
computeStride :: CSize -> Shape -> Stride
computeStride sizeOfElem shape =
  case V.length shape of {len ->
    V.constructrN len (
      \ accum ->
        if V.null accum then
          fromIntegral sizeOfElem
        else fromIntegral (shape ! (len - V.length accum)) * V.head accum
    )
  }

-- | Validate index correctness.
--
--   Signature: @shape -> index -> isValid@
validateIndex :: Shape -> Vector Int -> Bool
validateIndex shape index
  | V.length shape /= V.length index                   = False
  | V.any (< 0) index                                  = False
  | V.or $ V.zipWith ((>=) . fromIntegral) index shape = False
  | otherwise                                          = True

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
normalizeIndex :: Shape -> Vector Int -> Vector Int
normalizeIndex = V.zipWith normalizeItem

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
  Shape -> Shape -> Stride -> Stride -> (Shape, Shape, Stride, Stride)
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

-- | Parse -1 in new shape in @view@.
--
--   Signature: @shape -> newShape -> parsedShape@
parseNewShape :: Shape -> Index -> Shape
parseNewShape shape newShape =
  case elemIndices (-1) newShape of
    [] -> indexToShape newShape
    [dim] ->
      V.fromList
      $ map (
        \ dim ->
          if dim == -1 then
            case foldl' (*) (-1) newShape of {newTotalElems ->
              if newTotalElems > 0 then
                totalElems_ shape `div` fromIntegral newTotalElems
              else
                error
                $ "cannot reshape tensor of shape "
                ++ show shape
                ++ " into shape "
                ++ show newShape
            }
          else if dim < 0 then
            error
            $ "invalid tensor shape "
            ++ show newShape
          else fromIntegral dim
      ) newShape
    _ ->
      error
      $ "can only specify one unknown dimension "
      ++ show newShape

-- | Determine if all elements in a list are equal.
allEqual :: Eq a => [a] -> Bool
allEqual xs = all (== head xs) $ tail xs

-- | Swap index dimensions.
--
--   Signature: @index -> dim1 -> dim2 -> swappedIndex@
swapElementsAt :: (Storable t) => Vector t -> Int -> Int -> Vector t
swapElementsAt index dim1 dim2 =
  index // [(dim1, index ! dim2), (dim2, index ! dim1)]

{-# INLINE shapeToIndex #-}
{-# INLINE totalElems #-}
{-# INLINE totalElems_ #-}
{-# INLINE computeStride #-}
{-# INLINE validateIndex #-}
{-# INLINE normalizeItem #-}
{-# INLINE normalizeIndex #-}
{-# INLINE verifyBroadcastable #-}
{-# INLINE broadcastShapesStrides #-}
{-# INLINE broadcastedShape #-}
{-# INLINE parseNewShape #-}
{-# INLINE allEqual #-}
{-# INLINE swapElementsAt #-}
