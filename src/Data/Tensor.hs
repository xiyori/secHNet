{-# LANGUAGE InstanceSigs #-}
module Data.Tensor where

import Control.Applicative
import Data.Array (array, listArray, elems, bounds, Array, (!))
import qualified Data.Array as A
import Data.Matrix (matrix, Matrix, (!))
import qualified Data.Matrix as M
import Data.Index

data Tensor t = Tensor {
  shape :: Index,
  tensorData :: Array Int (Matrix t)
}

instance Functor Tensor where
  fmap :: (a -> b) -> Tensor a -> Tensor b
  fmap f (Tensor shape dat) =
    Tensor shape $ fmap (fmap f) dat

instance (A.Ix i, Num i) => Applicative (Array i) where
  pure :: (A.Ix i, Num i) => a -> Array i a
  pure elem = array (1, 1) [(1, elem)]

  liftA2 :: A.Ix i =>
    (a -> b -> c) -> Array i a -> Array i b -> Array i c
  liftA2 f x1 x2 =
    listArray (bounds x1)
    $ zipWith f (elems x1) (elems x2)

instance Applicative Tensor where
  pure :: a -> Tensor a
  pure elem = tensor [1] $ const elem

  liftA2 :: (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
  liftA2 f (Tensor shape dat1) (Tensor _ dat2) =
    Tensor shape $ liftA2 (liftA2 f) dat1 dat2

instance Foldable Tensor where
  foldMap :: Monoid m => (a -> m) -> Tensor a -> m
  foldMap f x@(Tensor shape dat) =
    foldMap (foldMap f) dat

instance (Num t) => Num (Tensor t) where
  (+) :: Num t => Tensor t -> Tensor t -> Tensor t
  (+) = _performWithBroadcasting (+)

  (-) :: Num t => Tensor t -> Tensor t -> Tensor t
  (-) = _performWithBroadcasting (-)

  (*) :: Num t => Tensor t -> Tensor t -> Tensor t
  (*) = _performWithBroadcasting (*)

  abs :: Num t => Tensor t -> Tensor t
  abs = fmap abs

  signum :: Num t => Tensor t -> Tensor t
  signum = fmap signum

  fromInteger :: Num t => Integer -> Tensor t
  fromInteger = pure . fromInteger

instance (Fractional t) => Fractional (Tensor t) where
  (/) :: Fractional t => Tensor t -> Tensor t -> Tensor t
  (/) = _performWithBroadcasting (/)

  fromRational :: Fractional t => Rational -> Tensor t
  fromRational = pure . fromRational

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
flatten x@(Tensor shape _) =
  tensor [numel x] $ (x !?) . fromInt shape . head

mean :: Fractional t => Tensor t -> t
mean x = sum x / numel x

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

(@) :: Num t => Tensor t -> Tensor t -> Tensor t
(@) = dot

_performWithBroadcasting ::
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
_performWithBroadcasting f x1@(Tensor shape1 _) x2@(Tensor shape2 _)
  | shape1 == [1] = fmap (f $ x1 !? shape1) x2
  | shape2 == [1] = fmap (flip f $ x2 !? shape2) x1
  | otherwise     = uncurry (liftA2 f) $ broadcast x1 x2

broadcast :: Tensor a -> Tensor b -> (Tensor a, Tensor b)
broadcast x1 x2 = (x1, x2)
