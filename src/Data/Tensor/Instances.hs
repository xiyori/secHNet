module Data.Tensor.Instances where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import Data.List
import Data.Tensor.Index
import Data.Tensor.Definitions
import Data.Tensor.Functional as T

import Foreign.C.Types


instance Storable t => Eq (Tensor t) where
  (==) = tensorEqual

instance NumTensor t => Num (Tensor t) where
  (+) = numTAdd
  (-) = numTSub
  (*) = numTMult
  negate = numTNegate
  abs = numTAbs
  signum = numTSignum
  fromInteger = scalar . fromInteger

instance FractionalTensor t => Fractional (Tensor t) where
  (/) = fracTDiv
  fromRational = scalar . fromRational

instance FloatingTensor t => Floating (Tensor t) where
  pi = scalar pi
  exp = floatTExp
  log = floatTLog
  sin = floatTSin
  cos = floatTCos
  asin = floatTAsin
  acos = floatTAcos
  atan = floatTAtan
  sinh = floatTSinh
  cosh = floatTCosh
  asinh = floatTAsinh
  acosh = floatTAcosh
  atanh = floatTAtanh

instance (Storable t, HasDtype t, Show t) => Show (Tensor t) where
  show x@(Tensor shape _ _ _)
    -- Print info about empty tensor
    | totalElems shape == 0 =
      "tensor([], shape="
      ++ show shape
      ++ ", dtype="
      ++ tensorDtype x
      ++ ")"
    -- Print all elements
    | totalElems shape <= maxElements =
      "tensor("
      ++ goAll (maxLengthAll []) "       " []
      ++ ")"
    -- Print first and last 3 elements of each dim
    | otherwise =
      "tensor("
      ++ goPart (maxLengthPart []) "       " []
      ++ ")"
    where
      nDims = V.length shape
      maxLine = 70
      maxElements = 1000
      nPart = 3
      ldots = -(nPart + 1)

      showCompact x =
        let s = show x in
          if ".0" `isSuffixOf` s then
            init s
          else s

      maxLengthAll index
        | dim < nDims =
          maximum $ Prelude.map (maxLengthAll . (
            \ i -> index ++ [i]
          )) [0 .. (shape V.! dim) - 1]
        | otherwise = length $ showCompact $ x ! index
        where
          dim = length index

      goAll fLength prefix index
        | dim < nDims =
          (if not (null index) && last index /= 0 then
            ","
            ++ replicate (nDims - dim) '\n'
            ++ prefix
          else "")
          ++ "["
          ++ concatMap (goAll fLength (prefix ++ " ") . (
            \ i -> index ++ [i]
          )) [0 .. (shape V.! dim) - 1]
          ++ "]"
        | otherwise =
          let strElem = showCompact (x ! index)
              maxLineIndex = fromIntegral $
                (maxLine - length prefix) `div` (fLength + 2) in
            (if not (null index) && last index > 0 &&
                last index `mod` maxLineIndex == 0 then
              ",\n" ++ prefix
            else if not (null index) && last index > 0 then
              ", "
            else "")
            ++ replicate (fLength - length strElem) ' '
            ++ strElem
        where
          dim = length index

      maxLengthPart index
        | dim < nDims =
          maximum $ Prelude.map (maxLengthPart . (
            \ i -> index ++ [i]
          )) $
          if shape V.! dim > nPart * 2 then
            [0 .. nPart - 1] ++ [-nPart .. -1]
          else [0 .. (shape V.! dim) - 1]
        | otherwise = length $ showCompact $ x ! index
        where
          dim = length index

      goPart fLength prefix index
        | not (null index) && last index == ldots =
          if dim == nDims then
            ", ..."
          else
            ","
            ++ replicate (nDims - dim) '\n'
            ++ prefix
            ++ "..."
        | dim < nDims =
          (if not (null index) && last index /= 0 then
            ","
            ++ replicate (nDims - dim) '\n'
            ++ prefix
          else "")
          ++ "["
          ++ concatMap (goPart fLength (prefix ++ " ") . (
            \ i -> index ++ [i]
          )) (
            if shape V.! dim > nPart * 2 then
              [0 .. nPart - 1] ++ [ldots] ++ [-nPart .. -1]
            else
              [0 .. (shape V.! dim) - 1]
          ) ++ "]"
        | otherwise =
          let strElem = showCompact (x ! index)
              normI = normalizeItem (2 * nPart) (last index)
              maxLineIndex = fromIntegral $
                (maxLine - length prefix) `div` (fLength + 2) in
            -- show normI ++ " " ++ show maxLineIndex ++ " " ++
            (if normI > 0 && normI `mod` maxLineIndex == 0 then
              ",\n" ++ prefix
            else if normI > 0 then
              ", "
            else "")
            ++ replicate (fLength - length strElem) ' '
            ++ strElem
        where
          dim = length index
  {-# INLINE show #-}
