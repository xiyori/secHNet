module Data.Tensor where

import Control.Applicative
import Data.Array (array, Array, (!))
import qualified Data.Array as A
import Data.Matrix (matrix, Matrix, (!))
import qualified Data.Matrix as M
import Data.Index

data Tensor t = Tensor {
  shape :: Index,
  tensorData :: Array Int (Matrix t)
}

tensor :: Index -> (Index -> t) -> Tensor t
tensor shape builder =
  Tensor shape
  $ array (1, toInt arrayShape arrayShape)
  $ map (
    \index -> (
      toInt arrayShape index,
      matrix matrixRows matrixCols $ builder . mergeIndex index
    ))
  $ indexRange0 arrayShape
  where
    (arrayShape, (matrixRows, matrixCols)) = splitIndex shape

full :: Index -> t -> Tensor t
full shape value = tensor shape $ const value

zeros :: Num t => Index -> Tensor t
zeros shape = full shape 0

ones :: Num t => Index -> Tensor t
ones shape = full shape 1

eye :: Num t => Index -> Tensor t
eye shape =
  tensor shape (
    \index ->
      if all (== head index) $ tail index then
        1
      else 0
  )

getElem :: Tensor t -> Index -> t
getElem (Tensor shape dat) index = (dat A.! arrayIndex) M.! matrixIndex
  where
    (arrayIndex, matrixIndex) = toInternal shape index

(!?) :: Tensor t -> Index -> t
(!?) = getElem

getElems :: Tensor t -> MultiIndex -> Tensor t
getElems x@(Tensor shape _) multiIndex =
  tensor newShape (\index -> x !? liftA2 (!!) expandedIndex index)
  where
    expandedIndex =
      zipWith (
        \dim slice ->
          if null slice then
            [1 .. dim]
          else slice
      ) shape multiIndex
    newShape = map length expandedIndex

numel :: Tensor t -> Int
numel (Tensor shape _) = countIndex shape

transpose :: Tensor t -> Tensor t
transpose (Tensor shape dat) =
  Tensor shape $ fmap M.transpose dat

flatten :: Tensor t -> Tensor t
flatten x@(Tensor shape dat) =
  tensor [numel x] $ (x !?) . fromInt shape . head

sum :: Num t => Tensor t -> t
sum (Tensor _ dat) =
  foldl (\accum mat -> accum + Prelude.sum mat) 0 dat

swapdim :: Tensor t -> Int -> Int -> Tensor t
swapdim x@(Tensor shape _) from to =
  tensor (swapElementsAt from to shape) ((x !?) . swapElementsAt from to)

dot :: Num t => Tensor t -> Tensor t -> Tensor t
dot (Tensor shape1 dat1) (Tensor shape2 dat2) =
  Tensor (mergeIndex arrayShape (matrixRows1, matrixCols2))
  $ array (1, toInt arrayShape arrayShape)
  $ map ((
    \index -> (
      index,
      (dat1 A.! index) * (dat2 A.! index)
    )) . toInt arrayShape)
  $ indexRange0 arrayShape
  where
    (arrayShape, (matrixRows1, _)) = splitIndex shape1
    (_, (_, matrixCols2)) = splitIndex shape2

