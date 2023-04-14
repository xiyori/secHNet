{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Tensor.FloatTensor where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
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
-- Include C headers.
C.include "<math.h>"
-- Include C utils.
C.include "cbits/float_tensor.h"


instance NumTensor CFloat where
  eye :: CInt -> CInt -> CInt -> Tensor CFloat
  eye rows columns diagonalIndex =
    Tensor (V.fromList [rows, columns])
    (V.fromList [columns * sizeOfCFloat, sizeOfCFloat]) 0
    $ unsafePerformIO
    $ do
      mutableData <- VM.new $ fromIntegral $ rows * columns
      VM.unsafeWith mutableData (
        \ mutableDataPtr ->
          [CU.exp| void {
            eye_f(
              $(int rows),
              $(int columns),
              $(int diagonalIndex),
              $(float *mutableDataPtr)
            )
          } |]
        )
      V.unsafeFreeze mutableData

  arange :: CFloat -> CFloat -> CFloat -> Tensor CFloat
  arange low high step =
    tensor [floor $ (high - low) / step] (
      \ fIndex -> low + step * fromIntegral fIndex
    )

  sum :: Tensor CFloat -> CFloat
  sum (Tensor shape stride offset dat) =
    case V.unsafeCast dat of {dataCChar ->
      [CU.pure| float {
        sum_f(
          $vec-len:shape,
          $vec-ptr:(int *shape),
          $vec-ptr:(int *stride),
          $(int offset),
          $vec-ptr:(char *dataCChar)
        )
      } |]
    }

  numTAdd :: Tensor CFloat -> Tensor CFloat -> Tensor CFloat
  numTAdd x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            elementwise_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride1),
              $(int offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(int *stride2),
              $(int offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar),
              add_f
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}

  numTSub :: Tensor CFloat -> Tensor CFloat -> Tensor CFloat
  numTSub x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            elementwise_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride1),
              $(int offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(int *stride2),
              $(int offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar),
              sub_f
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}

  numTMult :: Tensor CFloat -> Tensor CFloat -> Tensor CFloat
  numTMult x1 x2 =
    case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (sizeOfElem dat1) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            elementwise_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride1),
              $(int offset1),
              $vec-ptr:(char *data1CChar),
              $vec-ptr:(int *stride2),
              $(int offset2),
              $vec-ptr:(char *data2CChar),
              $vec-ptr:(char *mutableDataCChar),
              mult_f
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}

  numTNegate :: Tensor CFloat -> Tensor CFloat
  numTNegate (Tensor shape stride offset dat) =
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            map_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride),
              $(int offset),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar),
              neg_f
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }

  numTAbs :: Tensor CFloat -> Tensor CFloat
  numTAbs (Tensor shape stride offset dat) =
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            map_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride),
              $(int offset),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar),
              fabsf
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }

  numTSignum :: Tensor CFloat -> Tensor CFloat
  numTSignum (Tensor shape stride offset dat) =
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            map_f(
              $vec-len:shape,
              $vec-ptr:(int *shape),
              $vec-ptr:(int *stride),
              $(int offset),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar),
              sign_f
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }
  {-# INLINE eye #-}
  {-# INLINE arange #-}
  {-# INLINE sum #-}
  {-# INLINE numTAdd #-}
  {-# INLINE numTSub #-}
  {-# INLINE numTMult #-}
  {-# INLINE numTNegate #-}
  {-# INLINE numTAbs #-}
  {-# INLINE numTSignum #-}
