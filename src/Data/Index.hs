module Data.Index where

import Data.Vector.Unboxed (generate, fromList, constructN, constructrN,
                            Unbox, Vector, (!), (//))
import qualified Data.Vector.Unboxed as V
import Data.List (foldl')

type Index = Vector Int
type Slices = [Vector Int]

-- | Parse negative index value.
normalizeItem :: Int -> Int -> Int
normalizeItem dim i =
  if i < 0 then
    dim + i
  else i

-- | Parse negative index values.
normalizeIndex :: Index -> Index -> Index
normalizeIndex = V.zipWith normalizeItem

-- | Convert tensor index to internal representation.
-- toInternal :: Index -> Index -> (Int, MIndex)
-- toInternal shape index =
--   (toInt (fst $ splitIndex shape) arrayIndex, matrixIndex)
--   where
--     (arrayIndex, matrixIndex) = splitIndex index

-- | Flatten tensor index to integer index.
flattenIndex :: Index -> Index -> Int
flattenIndex shape index =
  V.sum $ V.zipWith (*) (_flatCoeffs shape) index

-- | Convert flattened integer index to tensor index.
unravelIndex :: Index -> Int -> Index
unravelIndex shape i =
  V.zipWith div (V.map (i `mod`) unravelCoeffs) flatCoeffs
  where
    len = V.length shape
    flatCoeffs = _flatCoeffs shape
    unravelCoeffs = _unravelCoeffs shape
  -- snd . V.foldl' (
  --   \ (i, accum) coeff -> (i `mod` coeff, (i `div` coeff) : accum)
  -- ) (index, [])
  -- $ _dimCoeffs shape

-- | Coefficients for index flattening.
_flatCoeffs :: Index -> Index
_flatCoeffs shape =
  constructrN len (
    \ accum ->
      if V.null accum then
        1
      else shape ! (len - V.length accum) * V.head accum
  )
  where
    len = V.length shape

-- | Coefficients for index flattening.
_unravelCoeffs :: Index -> Index
_unravelCoeffs shape =
  constructrN len (
    \ accum ->
      if V.null accum then
        V.last shape
      else shape ! (len - V.length accum - 1) * V.head accum
  )
  where
    len = V.length shape

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

-- | Total number of elements with indices between 1 and high.
countIndex :: Index -> Int
countIndex = V.foldl' (*) 1

-- | Validate index correctness.
validateIndex :: Index -> Index -> Bool
validateIndex shape index
  | V.length shape /= V.length index = False
  | V.or $ V.zipWith (>=) index shape = False
  | V.any (< 0) index                = False
  | otherwise                        = True

-- | Swap index dimensions.
swapElementsAt :: Int -> Int -> Index -> Index
swapElementsAt i j index =
  index // [(i, index ! j), (j, index ! i)]

allEqual :: Eq a => [a] -> Bool
allEqual xs = all (== head xs) $ tail xs

allEqualV :: (Unbox t, Eq t) => Vector t -> Bool
allEqualV xs = V.all (== V.head xs) xs

{-# INLINE normalizeItem #-}
{-# INLINE normalizeIndex #-}
{-# INLINE flattenIndex #-}
{-# INLINE unravelIndex #-}
{-# INLINE _flatCoeffs #-}
{-# INLINE _unravelCoeffs #-}
{-# INLINE countIndex #-}
{-# INLINE validateIndex #-}
{-# INLINE swapElementsAt #-}
{-# INLINE allEqual #-}
{-# INLINE allEqualV #-}
