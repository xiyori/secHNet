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

instance (Storable t, Show t) => Show (Tensor t) where
  show x@(Tensor shape _ _ _)
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
          "["
          ++ concatMap (goAll fLength (prefix ++ " ") . (
            \ i -> index ++ [i]
          )) [0 .. (shape V.! dim) - 1]
          ++
            if not (null index) && last index < (shape V.! (dim - 1)) - 1 then
              "],"
              ++ replicate (nDims - dim) '\n'
              ++ prefix
            else "]"
        | otherwise =
          let strElem = showCompact (x ! index)
              currentLine = (fLength + 2) * fromIntegral (last index + 1)
              previousLine = (fLength + 2) * fromIntegral (last index) in
            replicate (fLength - length strElem) ' '
            ++ strElem
            ++
              if not (V.null shape) &&
                 last index < V.last shape - 1 &&
                 currentLine `div` (maxLine - length prefix) >
                 previousLine `div` (maxLine - length prefix) then
                ",\n" ++ prefix
              else if not (V.null shape) &&
                      last index < V.last shape - 1 then
                ", "
              else ""
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
            "..., "
          else
            "...,"
            ++ replicate (nDims - dim) '\n'
            ++ prefix
        | dim < nDims =
          let normI = normalizeItem (2 * nPart) (last index) in
            "["
            ++ concatMap (goPart fLength (prefix ++ " ") . (
              \ i -> index ++ [i]
            )) (
              if shape V.! dim > nPart * 2 then
                [0 .. nPart - 1] ++ [ldots] ++ [-nPart .. -1]
              else
                [0 .. (shape V.! dim) - 1]
            ) ++
              if not (null index) &&
                 normI < (shape V.! (dim - 1)) - 1 &&
                 normI < 2 * nPart - 1 then
                "],"
                ++ replicate (nDims - dim) '\n'
                ++ prefix
              else "]"
        | otherwise =
          let strElem = showCompact (x ! index)
              normI = normalizeItem (2 * nPart) (last index)
              currentLine = (fLength + 2) * fromIntegral (normI + 1)
              previousLine = (fLength + 2) * fromIntegral normI in
            replicate (fLength - length strElem) ' '
            ++ strElem
            ++
              if not (V.null shape) &&
                 normI < V.last shape - 1 && normI < 2 * nPart - 1 &&
                 currentLine `div` (maxLine - length prefix) >
                 previousLine `div` (maxLine - length prefix) then
                ",\n" ++ prefix
              else if not (V.null shape) &&
                      normI < V.last shape - 1 && normI < 2 * nPart - 1 then
                ", "
              else ""
        where
          dim = length index
  {-# INLINE show #-}
