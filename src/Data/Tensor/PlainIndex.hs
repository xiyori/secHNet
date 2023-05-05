module Data.Tensor.PlainIndex where

import Data.Vector.Storable (Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.List
import Data.Maybe
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
parseIndex :: Shape -> Vector Int -> Shape
parseIndex shape index =
  case V.length shape of {nDims ->
    if V.length index == nDims then
      V.zipWith3 (
        \ axis dim i ->
          case normalizeItem dim i of {normI ->
            if 0 <= normI && normI < fromIntegral dim then
              fromIntegral normI
            else
              error
              $ "index "
              ++ show i
              ++ " is out of bounds for dim "
              ++ show axis
              ++ " with size "
              ++ show dim
          }
      ) (V.fromList [0 .. nDims - 1]) shape index
    else
      error
      $ "too many indices for tensor: tensor is "
      ++ show nDims
      ++ "-dimensional, but "
      ++ show (V.length index)
      ++ " were indexed"
  }

-- | Determine if all elements in a list are equal.
allEqual :: Eq a => [a] -> Bool
allEqual [] = True
allEqual xs = all (== head xs) $ tail xs

-- | Determine if shapes can be broadcasted.
--
--   Signature: @shapes -> canBroadcast@
verifyBroadcastable :: [Shape] -> Bool
verifyBroadcastable shapes =
  go $ map V.reverse shapes
    where
      go shapes
        | any V.null shapes = True
        | otherwise = allEqual (filter (/= 1) $ map V.head shapes) &&
                      go (map V.tail shapes)

-- | Broadcast shapes and strides.
--
--   Signature: @shapes -> strides -> (shape, strides)@
broadcastShapesStrides :: [Shape] -> [Stride] -> (Shape, [Stride])
broadcastShapesStrides shapes strides =
  case foldl' (
    \ accum@(maxLen, _) shape ->
      if V.length shape > maxLen then
        (V.length shape, shape)
      else accum
  ) (0, V.empty) shapes of {(nDims, maxShape) ->
  case map (
    \ shape ->
      V.take (nDims - V.length shape) maxShape
      V.++ shape
  ) shapes of {expandedShapes ->
    (V.generate (V.length $ head expandedShapes) (
      \ dim ->
        fromMaybe 1
        $ find (/= 1)
        $ map (V.! dim) expandedShapes
    ),
    zipWith (
      \ shape stride ->
        V.replicate (nDims - V.length stride) 0
        V.++ V.zipWith (
          \ dim stride ->
            if dim == 1 then
              0
            else stride
        ) shape stride
    ) shapes strides)
  }}

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

-- | Parse negative index value.
--
--   Signature: @dim -> i -> normI@
normalizeNewDim :: (Integral a, Integral b) => a -> b -> b
normalizeNewDim nDims dim =
  if dim < 0 then
    fromIntegral nDims + 1 + dim
  else dim

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
{-# INLINE normalizeItem #-}
{-# INLINE parseIndex #-}
{-# INLINE allEqual #-}
{-# INLINE verifyBroadcastable #-}
{-# INLINE broadcastShapesStrides #-}
{-# INLINE parseNewShape #-}
{-# INLINE normalizeNewDim #-}
{-# INLINE swapElementsAt #-}
