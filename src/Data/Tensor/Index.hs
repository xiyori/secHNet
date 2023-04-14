module Data.Tensor.Index where

import Data.Vector.Storable (generate, fromList, constructN, constructrN,
                             Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.List (foldl')
import System.IO.Unsafe
import Foreign
import Foreign.C.Types

-- | Tensor index @Vector CInt@.
type Index = Vector CInt

-- | Coefficients for index flattening.
--
--   Signature: @sizeOfElem -> shape -> stride@
computeStride :: CInt -> Index -> Index
computeStride sizeOfElem shape =
  case V.length shape of {len ->
    constructrN len (
      \ accum ->
        if V.null accum then
          sizeOfElem
        else shape ! (len - V.length accum) * V.head accum
    )
  }

-- | Parse negative index value.
--
--   Signature: @dim -> i -> normI@
normalizeItem :: (Num t, Ord t) => t -> t -> t
normalizeItem dim i =
  if i < 0 then
    dim + i
  else i

-- | Parse negative index values.
--
--   Signature: @shape -> index -> normIndex@
normalizeIndex :: Index -> Index -> Index
normalizeIndex = V.zipWith normalizeItem

-- | Total number of elements in a tensor with shape @shape@.
--
--   Signature: @shape -> numel@
totalElems :: Index -> CInt
totalElems = V.foldl' (*) 1

-- | Validate index correctness.
--
--   Signature: @shape -> index -> isValid@
validateIndex :: Index -> Index -> Bool
validateIndex shape index
  | V.length shape /= V.length index  = False
  | V.or $ V.zipWith (>=) index shape = False
  | V.any (< 0) index                 = False
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

-- | Determine if all elements in a list are equal.
allEqual :: Eq a => [a] -> Bool
allEqual xs = all (== head xs) $ tail xs

-- | Swap index dimensions.
--
--   Signature: @index -> dim1 -> dim2 -> swappedIndex@
swapElementsAt :: Index -> Int -> Int -> Index
swapElementsAt index dim1 dim2 =
  index // [(dim1, index ! dim2), (dim2, index ! dim1)]

-- | Generate all indices between low and high (inclusive).
-- indexRange :: Index -> Index -> [Index]
-- indexRange low high = reverse $ go [low]
--   where
--     go range@(lastIndex : _) =
--       if nextIndex == low then
--         range
--       else go (nextIndex : range)
--       where
--         nextIndex =
--           snd
--           $ foldr (
--             \ (l, h, i) (needAdd, index) ->
--               if needAdd && i == h then
--                 (True, l : index)
--               else if needAdd then
--                 (False, i + 1 : index)
--               else (False, i : index)
--           ) (True, [])
--           $ zip3 low high lastIndex

-- | Generate all indices between 1 and high (inclusive).
-- indexRange0 :: Index -> [Index]
-- indexRange0 high = indexRange (map (const 1) high) high

{-# INLINE computeStride #-}
{-# INLINE normalizeItem #-}
{-# INLINE normalizeIndex #-}
{-# INLINE totalElems #-}
{-# INLINE validateIndex #-}
{-# INLINE verifyBroadcastable #-}
{-# INLINE allEqual #-}
{-# INLINE swapElementsAt #-}
