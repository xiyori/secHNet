{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}

module Data.Tensor.Boolean where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.Tensor.PlainIndex
import Data.Tensor.Size
import Data.Tensor.Definitions
import Data.Tensor.Functional as T
import Data.Tensor.Instances

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx)
-- Include C utils.
C.include "cbits/boolean.h"


instance (HasDtype t, Eq t) => Eq (Tensor t) where
  (==) = equal

-- | Return True if two tensors have the same shape and elements.
equal :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Bool
equal (Tensor shape stride1 offset1 dat1)
      (Tensor shape2 stride2 offset2 dat2)
  | shape Prelude.== shape2 =
    case sizeOfElem dat1 of {elemSize ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      toBool [CU.pure| int {
        equal(
          $vec-len:shape,
          $vec-ptr:(size_t *shape),
          $(size_t elemSize),
          $vec-ptr:(long long *stride1),
          $(size_t offset1),
          $vec-ptr:(char *data1CChar),
          $vec-ptr:(long long *stride2),
          $(size_t offset2),
          $vec-ptr:(char *data2CChar)
        )
      } |]
    }}}
  | otherwise = False

infix 4 `equal`

-- | Return True if two arrays are element-wise equal within a tolerance.
allCloseTol :: (HasDtype t, Floating t) => CDouble -> CDouble -> Tensor t -> Tensor t -> Bool
allCloseTol rtol atol
  x1@(Tensor shape stride1 offset1 dat1)
     (Tensor shape2 stride2 offset2 dat2)
  | shape Prelude.== shape2 =
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      toBool [CU.pure| int {
        tensor_allclose(
          $(double rtol),
          $(double atol),
          $vec-len:shape,
          $vec-ptr:(size_t *shape),
          $(int dtype),
          $vec-ptr:(long long *stride1),
          $(size_t offset1),
          $vec-ptr:(char *data1CChar),
          $vec-ptr:(long long *stride2),
          $(size_t offset2),
          $vec-ptr:(char *data2CChar)
        )
      } |]
    }}}
  | otherwise = False

-- | Return True if two arrays are element-wise equal within a default tolerance.
allClose :: (HasDtype t, Floating t) => Tensor t -> Tensor t -> Bool
allClose = allCloseTol 1e-05 1e-08

infix 4 `allClose`

-- | Return (x1 == x2) element-wise.
(==) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(==) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_equal(
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

infix 4 ==

-- | Return (x1 /= x2) element-wise.
(/=) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(/=) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_not_equal(
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

infix 4 /=

-- | Return (x1 > x2) element-wise.
(>) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(>) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_greater(
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

infix 4 >

-- | Return (x1 < x2) element-wise.
(<) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(<) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_less(
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

infix 4 <

-- | Return (x1 >= x2) element-wise.
(>=) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(>=) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_geq(
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

infix 4 >=

-- | Return (x1 <= x2) element-wise.
(<=) :: (HasDtype t, Eq t) => Tensor t -> Tensor t -> Tensor CBool
(<=) x1 x2 =
  case broadcast x1 x2 of {(
      Tensor shape stride1 offset1 dat1,
      Tensor _ stride2 offset2 dat2
    ) ->
    case tensorDtype x1 of {dtype ->
    case V.unsafeCast dat1 of {data1CChar ->
    case V.unsafeCast dat2 of {data2CChar ->
      Tensor shape (computeStride (fromIntegral $ sizeOf (undefined :: CBool)) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_leq(
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

infix 4 <=

-- | Invert boolean tensor.
not :: Tensor CBool -> Tensor CBool
not x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_not(
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

-- | Return (x1 & x2) element-wise.
(&) :: Tensor CBool -> Tensor CBool -> Tensor CBool
(&) = (*)

infixr 3 &

-- | Return (x1 | x2) element-wise.
(|.) :: Tensor CBool -> Tensor CBool -> Tensor CBool
(|.) = (+)

infixr 2 |.

{-# INLINE equal #-}
{-# INLINE allCloseTol #-}
{-# INLINE (==) #-}
{-# INLINE (/=) #-}
{-# INLINE (>) #-}
{-# INLINE (<) #-}
{-# INLINE (>=) #-}
{-# INLINE (<=) #-}
{-# INLINE not #-}
