module Data.Index where

import Data.List (foldl')

type Index = [Int]
type MIndex = (Int, Int)
type Slices = [[Int]]

-- | List getItem (!!) with indexing starting from 1.
(!*) :: [t] -> Int -> t
(!*) xs i = xs !! (i - 1)

-- | Separate last 2 elements of a tensor index.
splitIndex :: Index -> (Index, MIndex)
splitIndex = go []
  where
    go [] [] = error "empty index not allowed"
    go [] [i] = ([1], (i, 1))
    go [] [i, j] = ([1], (i, j))
    go first [i, j] = (reverse first, (i, j))
    go first (i : is) = go (i : first) is

-- | Convert array and matrix indices to tensor index.
mergeIndex :: Index -> MIndex -> Index
mergeIndex arrayIndex (i, j) = arrayIndex ++ [i, j]

-- | Parse negative index value.
normalizeItem :: Int -> Int -> Int
normalizeItem dim i =
  if i < 0 then
    dim + i + 1
  else i

-- | Parse negative index values.
normalizeIndex :: Index -> Index -> Index
normalizeIndex = zipWith normalizeItem

-- | Convert tensor index to internal representation.
toInternal :: Index -> Index -> (Int, MIndex)
toInternal shape index =
  (toInt (fst $ splitIndex shape) arrayIndex, matrixIndex)
  where
    (arrayIndex, matrixIndex) = splitIndex index

-- | Flatten tensor index to integer index.
toInt :: Index -> Index -> Int
toInt shape index =
  (+) 1
  $ sum
  $ zipWith (
    \ coeff i -> coeff * (i - 1)
  ) (_dimCoeffs shape) index

-- | Convert flattened integer index to tensor index.
fromInt :: Index -> Int -> Index
fromInt shape index =
  snd . foldl' (
    \ (i, accum) coeff -> (i `mod` coeff, ((i `div` coeff) + 1) : accum)
  ) (index - 1, [])
  $ _dimCoeffs shape

-- | Coefficients for index flattening.
_dimCoeffs :: [Int] -> [Int]
_dimCoeffs =
  foldr (\ dim coeffs@(coeff : _) -> dim * coeff : coeffs) [1] . tail

-- | Generate all indices between low and high (inclusive).
indexRange :: Index -> Index -> [Index]
indexRange low high = reverse $ go [low]
  where
    go range@(lastIndex : _) =
      if nextIndex == low then
        range
      else go (nextIndex : range)
      where
        nextIndex =
          snd
          $ foldr (
            \ (l, h, i) (needAdd, index) ->
              if needAdd && i == h then
                (True, l : index)
              else if needAdd then
                (False, i + 1 : index)
              else (False, i : index)
          ) (True, [])
          $ zip3 low high lastIndex

-- | Generate all indices between 1 and high (inclusive).
indexRange0 :: Index -> [Index]
indexRange0 high = indexRange (map (const 1) high) high

-- | Total number of elements with indices between 1 and high.
countIndex :: Index -> Int
countIndex = foldl' (*) 1

-- | Validate index correctness.
validateIndex :: Index -> Index -> Bool
validateIndex shape index
  | length shape /= length index = False
  | or $ zipWith (>) index shape = False
  | any (< 1) index              = False
  | otherwise                    = True

-- | Swap index dimensions.
swapElementsAt :: Int -> Int -> Index -> Index
swapElementsAt i j index = error "TODO use !*"
  -- left ++ [elemJ] ++ middle ++ [elemI] ++ right
  -- where
  --   elemI = index !! i
  --   elemJ = index !! j
  --   left = take i index
  --   middle = take (j - i - 1) (drop (i + 1) index)
  --   right = drop (j + 1) index

allEqual :: Eq a => [a] -> Bool
allEqual xs = all (== head xs) $ tail xs
