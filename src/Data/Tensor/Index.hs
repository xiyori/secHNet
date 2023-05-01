{-# LANGUAGE LambdaCase #-}

module Data.Tensor.Index where

import Data.Vector.Storable (Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.List
import System.IO.Unsafe
import Foreign
import Foreign.C.Types

-- | Tensor index @Vector CSize@.
type Index = Vector CSize

-- | Tensor stride @Vector CLLong@.
type Stride = Vector CLLong

-- | Slice data type.
data Slice
  -- | Single index @I index@.
  = I CLLong
  -- | Full slice, analogous to NumPy @:@.
  | A
  -- | Slice from start, analogous to NumPy @start:@.
  | S CLLong
  -- | Slice till end, analogous to NumPy @:end@.
  | E CLLong
  | CLLong  -- | Slice @start :. end@, analogous to
          --   NumPy @start:end@.
          :. CLLong
  | Slice -- | Slice @S start :| step@, @E end :| step@
          --   or @start :. end :| step@, analogous to
          --   NumPy @start:end:step@.
          :| CLLong
  -- | Insert new dim, analogous to NumPy @None@.
  | None
  -- | Ellipses, analogous to NumPy @...@.
  | Ell
  deriving (Eq, Show)

infixl 5 :.
infixl 5 :|

-- | Slice indexer data type.
type Slices = [Slice]

-- | Coefficients for index flattening.
--
--   Signature: @sizeOfElem -> shape -> stride@
computeStride :: CSize -> Index -> Stride
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
normalizeIndex :: Index -> Vector CLLong -> Vector CLLong
normalizeIndex = V.zipWith normalizeItem

-- | Total number of elements in a tensor with shape @shape@.
--
--   Signature: @shape -> numel@
totalElems :: Index -> CSize
totalElems = V.foldl' (*) 1

-- | Validate index correctness.
--
--   Signature: @shape -> index -> isValid@
validateIndex :: Index -> Vector CLLong -> Bool
validateIndex shape index
  | V.length shape /= V.length index  = False
  | V.any (< 0) index                 = False
  | V.or $ V.zipWith ((>=) . fromIntegral) index shape = False
  | otherwise                         = True

-- | Determine if two shapes can be broadcasted.
--
--   Signature: @shape -> shape -> canBroadcast@
verifyBroadcastable :: Index -> Index -> Bool
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
  Index -> Index -> Stride -> Stride -> (Index, Index, Stride, Stride)
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
broadcastedShape :: Index -> Index -> Index
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

-- | Parse @None@ in slices.
--
--   Signature: @nDims -> slices -> (dimsToInsert, parsedSlices)@
parseNone :: Int -> Slices -> ([Int], Slices)
parseNone nDims slices =
  case filter (
    \case
      I _ -> False
      _   -> True
  ) slices of {intervalsOnly ->
  case elemIndices Ell slices of
    [] ->
      (elemIndices None intervalsOnly,
       filter (/= None) slices)
    [dim] ->
      (elemIndices None (take dim intervalsOnly)
       ++ map (nDims - length intervalsOnly + dim +)
          (elemIndices None (drop dim intervalsOnly)),
       filter (/= None) slices)
    _ ->
      error "slices can only have a single ellipsis \'Ell\'"
  }

-- | Parse @Ell@ and pad slices to @nDims@ length.
--
--   Signature: @nDims -> slices -> parsedSlices@
parseEllipses :: Int -> Slices -> Slices
parseEllipses nDims slices =
  case elemIndices Ell slices of
    [] ->
      slices ++ replicate (nDims - length slices) A
    [dim] ->
      take dim slices
      ++ replicate (nDims - length slices + 1) A
      ++ drop (dim + 1) slices
    _ ->
      error "slices can only have a single ellipsis \'Ell\'"

-- | Parse negative slice values.
--
--   Signature: @i -> dim -> slice -> step -> normSlice@
normalizeSlice :: Int -> CLLong -> Slice -> CLLong -> Slice
normalizeSlice axis dim (I i) step =
  case normalizeItem dim i of {normI ->
    if 0 <= normI && normI < dim then
      I normI
    else
      error
      $ "index "
      ++ show i
      ++ " is out of bounds for dim "
      ++ show axis
      ++ " with size "
      ++ show dim
  }
normalizeSlice axis dim slice step
  | step > 0 =
    case slice of
      A -> 0 :. dim :| step
      S start ->
        case min dim $ max 0 $ normalizeItem dim start of {start ->
          start :. dim :| step
        }
      E end ->
        case min dim $ max 0 $ normalizeItem dim end of {end ->
          0 :. end :| step
        }
      start :. end ->
        case min dim $ max 0 $ normalizeItem dim start of {start ->
        case min dim $ max 0 $ normalizeItem dim end of {end ->
          start :. max start end :| step
        }}
      badSlice ->
        error $ "incorrect slice " ++ show (badSlice :| step)
  | step < 0 =
    case slice of
      A -> dim - 1 :. -1 :| step
      S start ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim start of {start ->
          start :. -1 :| step
        }
      E end ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim end of {end ->
          dim - 1 :. end :| step
        }
      start :. end ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim start of {start ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim end of {end ->
          max start end :. end :| step
        }}
      badSlice ->
        error $ "incorrect slice " ++ show (badSlice :| step)
  | otherwise =
    error "slice step cannot be zero"

-- | Parse slices and validate their correctness.
--
--   Signature: @shape -> slices -> parsedSlices@
parseSlices :: Index -> Slices -> ([Int], Slices)
parseSlices shape slices =
  case V.length shape of {nDims ->
  case parseNone nDims slices of {(dimsToInsert, slices) ->
  case parseEllipses nDims slices of {slices ->
    if length slices <= nDims then
      (dimsToInsert,
      zipWith3 (
        \ axis dim slice ->
          case slice of
            slice :| step ->
              normalizeSlice axis (fromIntegral dim) slice step
            _ ->
              normalizeSlice axis (fromIntegral dim) slice 1
      ) [0 .. V.length shape - 1]
      (V.toList shape) slices)
    else
      error
      $ "too many indices for tensor: tensor is "
      ++ show nDims
      ++ "-dimensional, but "
      ++ show (length slices)
      ++ " were indexed"
  }}}

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
{-# INLINE parseNone #-}
{-# INLINE parseEllipses #-}
{-# INLINE normalizeSlice #-}
{-# INLINE parseSlices #-}
