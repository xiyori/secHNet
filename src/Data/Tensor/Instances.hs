{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Tensor.Instances where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List
import Data.Tensor.Index
import Data.Tensor.Size
import Data.Tensor.Definitions
import Data.Tensor.Functional as T

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx)
-- Include C utils.
C.include "cbits/num_tensor.h"


instance HasDtype CBool where
  tensorDtype _ = 0
  showDtype _ = "bool"

instance HasDtype CChar where
  tensorDtype _ = 1
  showDtype _ = "int8"

instance HasDtype CUChar where
  tensorDtype _ = 2
  showDtype _ = "uint8"

instance HasDtype CShort where
  tensorDtype _ = 3
  showDtype _ = "int16"

instance HasDtype CUShort where
  tensorDtype _ = 4
  showDtype _ = "uint16"

instance HasDtype CInt where
  tensorDtype _ = 5
  showDtype _ = "int32"

instance HasDtype CUInt where
  tensorDtype _ = 6
  showDtype _ = "uint32"

instance HasDtype CLLong where
  tensorDtype _ = 7
  showDtype _ = "int64"

instance HasDtype CULLong where
  tensorDtype _ = 8
  showDtype _ = "uint64"

instance HasDtype CFloat where
  tensorDtype _ = 10
  showDtype _ = "float32"

instance HasDtype CDouble where
  tensorDtype _ = 11
  showDtype _ = "float64"

_rangeF :: (HasDtype t, RealFrac t) => t -> t -> t -> Tensor t
_rangeF low high step =
  tensor (V.singleton $ floor $ (high - low) / step) (
    \ fIndex -> low + step * fromIntegral fIndex
  )

_rangeI :: (HasDtype t, Integral t) => t -> t -> t -> Tensor t
_rangeI low high step =
  tensor (V.singleton $ fromIntegral $ (high - low) `div` step) (
    \ fIndex -> low + step * fromIntegral fIndex
  )

{-# INLINE _rangeF #-}
{-# INLINE _rangeI #-}

instance HasArange CChar where
  arange = _rangeI

instance HasArange CUChar where
  arange = _rangeI

instance HasArange CShort where
  arange = _rangeI

instance HasArange CUShort where
  arange = _rangeI

instance HasArange CInt where
  arange = _rangeI

instance HasArange CUInt where
  arange = _rangeI

instance HasArange CLLong where
  arange = _rangeI

instance HasArange CULLong where
  arange = _rangeI

instance HasArange CFloat where
  arange = _rangeF

instance HasArange CDouble where
  arange = _rangeF

instance (HasDtype t, Num t) => Num (Tensor t) where
  (+) x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_add(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $(int dtype),
              $vec-ptr:(long long *stride1),
              $(size_t offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(long long *stride2),
              $(size_t offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}}

  (-) x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_sub(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $(int dtype),
              $vec-ptr:(long long *stride1),
              $(size_t offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(long long *stride2),
              $(size_t offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}}

  (*) x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_mult(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $(int dtype),
              $vec-ptr:(long long *stride1),
              $(size_t offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(long long *stride2),
              $(size_t offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}}

  negate x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_neg(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  abs x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_abs(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  signum x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_sign(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  fromInteger = scalar . fromInteger
  {-# INLINE (+) #-}
  {-# INLINE (-) #-}
  {-# INLINE (*) #-}
  {-# INLINE negate #-}
  {-# INLINE abs #-}
  {-# INLINE signum #-}

instance (HasDtype t, Fractional t) => Fractional (Tensor t) where
  (/) x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_div(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $(int dtype),
              $vec-ptr:(long long *stride1),
              $(size_t offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(long long *stride2),
              $(size_t offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}}

  fromRational = scalar . fromRational
  {-# INLINE (/) #-}

instance (HasDtype t, Floating t) => Floating (Tensor t) where
  pi = scalar pi

  exp x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_exp(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  log x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_log(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  sin x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_sin(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  cos x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_cos(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  asin x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_asin(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  acos x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_acos(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  atan x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_atan(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  sinh x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_sinh(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  cosh x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_cosh(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  asinh x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_asinh(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  acosh x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_acosh(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}

  atanh x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_atanh(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}
  {-# INLINE exp #-}
  {-# INLINE log #-}
  {-# INLINE sin #-}
  {-# INLINE cos #-}
  {-# INLINE asin #-}
  {-# INLINE acos #-}
  {-# INLINE atan #-}
  {-# INLINE sinh #-}
  {-# INLINE cosh #-}
  {-# INLINE asinh #-}
  {-# INLINE acosh #-}

instance (HasDtype t, Show t) => Show (Tensor t) where
  show x@(Tensor shape _ _ _)
    -- Debug show
    -- | True =
    --   "tensor("
    --   ++ show (V.take 3 $ tensorData x)
    --   ++ ", shape="
    --   ++ show shape
    --   ++ ", stride="
    --   ++ show (tensorStride x)
    --   ++ ", offset="
    --   ++ show (tensorOffset x)
    --   ++ ", dtype="
    --   ++ showDtype x
    --   ++ ")"
    -- Print info about empty tensor
    | totalElems shape == 0 =
      "tensor([], shape="
      ++ show shape
      ++ ", dtype="
      ++ showDtype x
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
          )) [0 .. fromIntegral (shape V.! dim) - 1]
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
          )) [0 .. fromIntegral (shape V.! dim) - 1]
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
          if fromIntegral (shape V.! dim) > nPart * 2 then
            [0 .. nPart - 1] ++ [-nPart .. -1]
          else [0 .. fromIntegral (shape V.! dim) - 1]
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
            if fromIntegral (shape V.! dim) > nPart * 2 then
              [0 .. nPart - 1] ++ [ldots] ++ [-nPart .. -1]
            else
              [0 .. fromIntegral (shape V.! dim) - 1]
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
