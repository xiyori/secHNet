{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Data.Tensor.FloatTensor where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
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
C.include "cbits/float_tensor.h"


instance NumTensor CFloat where
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

  arange low high step =
    tensor (V.singleton $ floor $ (high - low) / step) (
      \ fIndex -> low + step * fromIntegral fIndex
    )

  sum (Tensor shape stride offset dat) =
    case V.unsafeCast dat of {dataCChar ->
      [CU.pure| float {
        sum_f(
          $vec-len:stride,
          $vec-ptr:(int *shape),
          $vec-ptr:(int *stride),
          $(int offset),
          $vec-ptr:(char *dataCChar)
        )
      } |]
    }

  numTPlus = error ""
  numTMinus = error ""
  numTMult = error ""
  numTAbs = error ""
  numTSignum = error ""
