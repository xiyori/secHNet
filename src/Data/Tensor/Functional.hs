{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Data.Tensor.Functional where

import Control.Monad.IO.Class
import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List
import System.Random
import Data.Random.Normal
import Data.Tensor.PlainIndex
import Data.Tensor.Size
import Data.Tensor.ListUtils
import Data.Tensor.Definitions as T

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx)
-- Include C utils.
C.include "cbits/core/core.h"
C.include "cbits/integral.h"
C.include "cbits/fold.h"
C.include "cbits/ord.h"
C.include "cbits/construct.h"
C.include "cbits/convert.h"
C.include "cbits/matmul.h"


-- Construction
-- ------------

-- | Convert a vector to a 1-D tensor.
--
--   Signature: @vector -> tensor@
fromVector :: HasDtype t => Vector t -> Tensor t
fromVector dat =
  case V.singleton $ fromIntegral $ V.length dat of {shape ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }

-- | Convert a list to a 1-D tensor.
--
--   Signature: @list -> tensor@
fromList :: HasDtype t => [t] -> Tensor t
fromList listData =
  case parseShape1 listData of {shape ->
  case parseData1 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Convert a list of lists to a 2-D tensor.
--
--   Signature: @list -> tensor@
fromList2 :: HasDtype t => [[t]] -> Tensor t
fromList2 listData =
  case parseShape2 listData of {shape ->
  case parseData2 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Convert a list of lists to a 3-D tensor.
--
--   Signature: @list -> tensor@
fromList3 :: HasDtype t => [[[t]]] -> Tensor t
fromList3 listData =
  case parseShape3 listData of {shape ->
  case parseData3 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Convert a list of lists to a 4-D tensor.
--
--   Signature: @list -> tensor@
fromList4 :: HasDtype t => [[[[t]]]] -> Tensor t
fromList4 listData =
  case parseShape4 listData of {shape ->
  case parseData4 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Convert a list of lists to a 5-D tensor.
--
--   Signature: @list -> tensor@
fromList5 :: HasDtype t => [[[[[t]]]]] -> Tensor t
fromList5 listData =
  case parseShape5 listData of {shape ->
  case parseData5 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Convert a list of lists to a 6-D tensor.
--
--   Signature: @list -> tensor@
fromList6 :: HasDtype t => [[[[[[t]]]]]] -> Tensor t
fromList6 listData =
  case parseShape6 listData of {shape ->
  case parseData6 listData of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Generate a tensor from a generator function.
--
--   /WARNING:/ This function can be very slow
--   for large tensors. Consider using other
--   constructors if possible.
--
--   Signature: @shape -> generator -> tensor@
tensor :: HasDtype t => Index -> (Index -> t) -> Tensor t
tensor shape builder =
  case indexToShape shape of {tShape ->
  case flattenCoeffs_ shape of {fCoeffs ->
  case unravelCoeffs_ shape of {uCoeffs ->
  case V.generate (fromIntegral $ totalElems_ tShape)
  $ builder . unravelIndex_ uCoeffs fCoeffs of {dat ->
    Tensor tShape (computeStride (sizeOfElem dat) tShape) 0 dat
  }}}}

-- | Generate a tensor from a generator function.
--
--   Signature: @shape -> generator -> tensor@
tensor_ :: HasDtype t => Shape -> (Int -> t) -> Tensor t
tensor_ shape builder =
  case V.generate (fromIntegral $ totalElems_ shape) builder of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }

-- | Return a new tensor filled with @fillValue@.
--
--   Signature: @shape -> fillValue -> tensor@
full :: HasDtype t => Index -> t -> Tensor t
full shape = full_ $ indexToShape shape

-- | Return a new tensor filled with @fillValue@.
--
--   Signature: @shape -> fillValue -> tensor@
full_ :: HasDtype t => Shape -> t -> Tensor t
full_ shape fillValue =
  if totalElems_ shape /= 0 then
    case V.singleton fillValue of {dat ->
      Tensor shape (V.replicate (V.length shape) 0) 0 dat
    }
  else
    case V.empty of {dat ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
    }

-- | Return a new tensor filled with zeros.
--
--   Signature: @shape -> tensor@
zeros :: (HasDtype t, Num t) => Index -> Tensor t
zeros shape = full shape 0

-- | Return a new tensor filled with ones.
--
--   Signature: @shape -> tensor@
ones :: (HasDtype t, Num t) => Index -> Tensor t
ones shape = full shape 1

-- | Return a new empty tensor of shape [0].
empty :: HasDtype t => Tensor t
empty = tensor_ (V.singleton 0) undefined

-- | Return a new tensor of shape [] with a single value.
scalar :: HasDtype t => t -> Tensor t
scalar = full_ V.empty

-- | Return a new tensor of shape [1] with a single value.
single :: HasDtype t => t -> Tensor t
single = full_ (V.singleton 1)

-- | Return a new tensor filled with random values from
--   uniform distribution [low, high].
--
--   Signature: @shape -> (low, high) -> gen -> (tensor, gen)@
randRange :: (HasDtype t, UniformRange t, RandomGen g) =>
  Index -> (t, t) -> g -> (Tensor t, g)
randRange shape (low, high) gen =
  case indexToShape shape of {shape ->
  case V.unfoldrExactN (fromIntegral $ totalElems_ shape)
  (uniformR (low, high)) gen of {dat ->
    (Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat,
    gen)
  }}

-- | Return a new tensor filled with random values from
--   uniform distribution [low, high].
--
--   Signature: @shape -> (low, high) -> tensor@
randRangeM :: (HasDtype t, UniformRange t, MonadIO m) =>
  Index -> (t, t) -> m (Tensor t)
randRangeM shape (low, high) =
  case indexToShape shape of {shape -> do
    dat <- V.replicateM (fromIntegral $ totalElems_ shape) (
      getStdGen >>= (
        \ gen ->
          case uniformR (low, high) gen of {(value, gen) ->
            setStdGen gen >> return value
          }
      ))
    return $ Tensor shape
      (computeStride (sizeOfElem dat) shape) 0 dat
  }

-- | Return a new tensor filled with random values from
--   uniform distribution [0, 1].
--
--   Signature: @shape -> gen -> (tensor, gen)@
rand :: (HasDtype t, UniformRange t, Num t, RandomGen g) =>
  Index -> g -> (Tensor t, g)
rand shape = randRange shape (0, 1)

-- | Return a new tensor filled with random values from
--   uniform distribution [0, 1].
--
--   Signature: @shape -> tensor@
randM :: (HasDtype t, UniformRange t, Num t, MonadIO m) =>
  Index -> m (Tensor t)
randM shape = randRangeM shape (0, 1)

-- | Return a new tensor filled with random values from
--   standard normal distribution.
--
--   Signature: @shape -> gen -> (tensor, gen)@
randn :: (HasDtype t, Random t, Floating t, RandomGen g) =>
  Index -> g -> (Tensor t, g)
randn shape gen =
  case indexToShape shape of {shape ->
  case V.unfoldrExactN (fromIntegral $ totalElems_ shape)
  normal gen of {dat ->
    (Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat,
     gen)
  }}

-- | Return a new tensor filled with random values from
--   standard normal distribution inside IO monad.
--
--   Signature: @shape -> tensor@
randnM :: (HasDtype t, Random t, Floating t, MonadIO m) =>
  Index -> m (Tensor t)
randnM shape =
  case indexToShape shape of {shape -> do
    dat <- V.replicateM (fromIntegral $ totalElems_ shape) (
      getStdGen >>= (
        \ gen ->
          case normal gen of {(value, gen) ->
            setStdGen gen >> return value
          }
      ))
    return $ Tensor shape
      (computeStride (sizeOfElem dat) shape) 0 dat
  }

-- | Generate a tensor from a generator function.
--
--   Signature: @tensor -> generator -> tensor@
tensorLike :: (HasDtype a, HasDtype b) => Tensor a -> (Index -> b) -> Tensor b
tensorLike x = tensor $ shape x

-- | Return a new tensor filled with @fillValue@.
--
--   Signature: @tensor -> fillValue -> tensor@
fullLike :: (HasDtype a, HasDtype b) => Tensor a -> b -> Tensor b
fullLike (Tensor shape _ _ _) = full_ shape

-- | Return a new tensor filled with zeros.
--
--   Signature: @tensor -> tensor@
zerosLike :: (HasDtype a, HasDtype b, Num b) => Tensor a -> Tensor b
zerosLike (Tensor shape _ _ _) = full_ shape 0

-- | Return a new tensor filled with ones.
--
--   Signature: @tensor -> tensor@
onesLike :: (HasDtype a, HasDtype b, Num b) => Tensor a -> Tensor b
onesLike (Tensor shape _ _ _) = full_ shape 1

-- | Return a 2-D tensor with ones
--   on the diagonal and zeros elsewhere.
--
--   @diagonalIndex@ 0 refers to the main diagonal,
--   a positive value refers to an upper diagonal,
--   and a negative value to a lower diagonal.
--
--   Signature: @rows -> columns -> diagonalIndex -> tensor@
eye :: forall t. (HasDtype t, Num t) => Int -> Int -> Int -> Tensor t
eye rows columns diagonalIndex
  | rows >= 0 && columns >= 0 =
    case (
      fromIntegral rows,
      fromIntegral columns,
      fromIntegral diagonalIndex
    ) of {(rows, columns, diagonalIndex) ->
    case empty :: Tensor t of {sampleTensor ->
    case tensorDtype sampleTensor of {dtype ->
    case V.fromList [rows, columns] of {shape ->
      Tensor shape (computeStride (sizeOfElem $ tensorData sampleTensor) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ rows * columns
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_eye(
              $(size_t rows),
              $(size_t columns),
              $(long long diagonalIndex),
              $(int dtype),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    }}}}
  | otherwise =
    error
    $ "invalid parameters in eye "
    ++ show (rows, columns, diagonalIndex)


-- Binary operations
-- -----------------

-- | Broadcast tensors without copying.
broadcast :: (HasDtype a, HasDtype b) =>
  Tensor a -> Tensor b -> (Tensor a, Tensor b)
broadcast (Tensor shape1 stride1 offset1 dat1)
          (Tensor shape2 stride2 offset2 dat2)
  | verifyBroadcastable [shape1, shape2] =
    case broadcastShapesStrides [shape1, shape2] [stride1, stride2]
    of {(shape, [stride1, stride2]) ->
      (Tensor shape stride1 offset1 dat1,
      Tensor shape stride2 offset2 dat2)
    }
  | otherwise =
    error
    $ "tensors of shapes "
    ++ show shape1
    ++ " "
    ++ show shape2
    ++ " can not be broadcasted"

-- | Broadcast tensors without copying.
broadcastN :: HasDtype t => [Tensor t] -> [Tensor t]
broadcastN xs =
  case Prelude.map tensorShape xs of {shapes ->
    if verifyBroadcastable shapes then
      case broadcastShapesStrides shapes $ Prelude.map tensorStride xs
      of {(shape, strides) ->
        zipWith (
          \ stride (Tensor _ _ offset dat) ->
            Tensor shape stride offset dat
        ) strides xs
      }
    else
      error
      $ "tensors of shapes "
      ++ unwords (Prelude.map show shapes)
      ++ " can not be broadcasted"
  }

-- | Broadcast tensor into new shape without copying.
--
-- Signature: @tensor -> shapeTo -> tensor@
broadcastTo :: HasDtype t => Tensor t -> Shape -> Tensor t
broadcastTo (Tensor shapeFrom stride offset dat) shapeTo
  | verifyBroadcastableTo shapeFrom shapeTo =
    case broadcastShapeStrideTo shapeFrom stride shapeTo
    of {stride ->
      Tensor shapeTo stride offset dat
    }
  | otherwise =
    error
    $ "can not broadcast tensor of shape "
    ++ show shapeFrom
    ++ " into shape "
    ++ show shapeTo

-- | Elementwise integer division truncated toward negative infinity.
(//) :: HasDtype t => Tensor t -> Tensor t -> Tensor t
(//) x1 x2 =
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
        mutableData <- VM.new $ fromIntegral $ totalElems_ shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_floor_div(
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

infixl 7 //

-- | Elementwise integer modulus, satisfying
--
-- > (x // y) * y + (x % y) == x
(%) :: HasDtype t => Tensor t -> Tensor t -> Tensor t
(%) x1 x2 =
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
        mutableData <- VM.new $ fromIntegral $ totalElems_ shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_mod(
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

infixl 7 %

-- | Make a tensor contiguous in last dim.
--
--   Helper function for @matmul@.
--
--   Signature: @tensor -> (performCblasTranspose, tensor)@
makeContiguousInLastDim :: HasDtype t => Tensor t -> (CInt, Tensor t)
makeContiguousInLastDim x@(Tensor shape stride offset dat) =
  case fromIntegral $ sizeOfElem dat of {elemSize ->
  case V.length shape of {nDims ->
    if V.last stride /= elemSize &&
        stride V.! (nDims - 2) /= elemSize then
      (fromBool False, copy x)
    else
      case (
        if V.last stride /= elemSize then
          (fromBool True,
          Tensor (swapElementsAt shape (nDims - 2, nDims - 1))
                 (swapElementsAt stride (nDims - 2, nDims - 1))
                 offset dat)
        else (fromBool False, x)
      ) of {(trans, x@(Tensor shape stride offset dat)) ->
        if stride V.! (nDims - 2) < fromIntegral (V.last shape) * elemSize then
          if shape V.! (nDims - 2) <= 1 then
            (trans,
            Tensor shape (stride V.// [(nDims - 2, elemSize)])
                   offset dat)
          else (fromBool False, copy x)
        else (trans, x)
      }
  }}

-- | Perform batch matrix multiplication.
--
--   Broadcasting rules follow NumPy matmul.
matmul :: (HasDtype t, Floating t) => Tensor t -> Tensor t -> Tensor t
matmul (Tensor shape1 stride1 offset1 dat1)
       (Tensor shape2 stride2 offset2 dat2) =
  case (V.length shape1, V.length shape2) of {(nDims1, nDims2) ->
  case elemIndex 0 [nDims1, nDims2] of
    Nothing ->
      case sizeOfElem dat1 of {elemSize ->
      case (
        if nDims1 == 1 then
          ([-2],
          2,
          V.cons 1 shape1,
          V.cons (fromIntegral $ V.head shape1 * elemSize) stride1)
        else ([], nDims1, shape1, stride1)
      ) of {(indicesToRemove, nDims1, shape1, stride1) ->
      case (
        if nDims2 == 1 then
          (-1 : indicesToRemove,
          2,
          V.snoc shape2 1,
          V.snoc stride2 (fromIntegral elemSize))
        else (indicesToRemove, nDims2, shape2, stride2)
      ) of {(indicesToRemove, nDims2, shape2, stride2) ->
      case (shape1 V.! (nDims1 - 2), V.last shape2, V.last shape1) of {(m, n, k) ->
        if shape2 V.! (nDims2 - 2) == k then
          case (
            V.take (nDims1 - 2) shape1,
            V.take (nDims2 - 2) shape2,
            V.take (nDims1 - 2) stride1,
            V.take (nDims2 - 2) stride2
          ) of {(batchShape1, batchShape2, batchStride1, batchStride2) ->
            if verifyBroadcastable [batchShape1, batchShape2] then
              case broadcastShapesStrides [batchShape1, batchShape2] [batchStride1, batchStride2]
              of {(batchShape, [batchStride1, batchStride2]) ->
              -- unsafePerformIO $ print (nDims1, shape1, stride1, nDims2, shape2, stride2) >> return (
              case makeContiguousInLastDim (Tensor shape1 stride1 offset1 dat1)
              of {(trans1, x1@(Tensor shape1 stride1 offset1 dat1)) ->
              case makeContiguousInLastDim (Tensor shape2 stride2 offset2 dat2)
              of {(trans2, x2@(Tensor shape2 stride2 offset2 dat2)) ->
              -- unsafePerformIO $ print (nDims1, shape1, stride1, nDims2, shape2, stride2) >> return (
              case (
                batchShape V.++ V.fromList [m, n],
                batchStride1 V.++ V.drop (nDims1 - 2) stride1,
                batchStride2 V.++ V.drop (nDims2 - 2) stride2
              ) of {(shape, stride1, stride2) ->
              case V.ifilter (
                \ axis dim -> axis - V.length shape `notElem` indicesToRemove
              ) shape of {newShape ->
              case tensorDtype x1 of {dtype ->
              case V.unsafeCast dat1 of {data1CChar ->
              case V.unsafeCast dat2 of {data2CChar ->
                Tensor newShape (computeStride elemSize newShape) 0
                $ unsafePerformIO
                $ do
                  mutableData <- VM.new $ fromIntegral $ totalElems_ newShape
                  case VM.unsafeCast mutableData of {mutableDataCChar ->
                    [CU.exp| void {
                      tensor_matmul(
                        $(size_t m),
                        $(size_t n),
                        $(size_t k),
                        $vec-len:shape,
                        $vec-ptr:(size_t *shape),
                        $(int dtype),
                        $(int trans1) ? CblasTrans : CblasNoTrans,
                        $vec-ptr:(long long *stride1),
                        $(size_t offset1),
                        $vec-ptr:(char *data1CChar),
                        $(int trans2) ? CblasTrans : CblasNoTrans,
                        $vec-ptr:(long long *stride2),
                        $(size_t offset2),
                        $vec-ptr:(char *data2CChar),
                        $vec-ptr:(char *mutableDataCChar)
                      )
                    } |]
                  }
                  V.unsafeFreeze mutableData
              }}}}}}}}
            else
              error
              $ "matmul: batch shapes "
              ++ show shape1
              ++ " "
              ++ show shape2
              ++ " can not be broadcasted"
          }
        else
          error
          $ "matmul: input operand has a mismatch in dim "
          ++ show (nDims2 - 2)
          ++ " (size "
          ++ show (shape2 V.! (nDims2 - 2))
          ++ " is different from "
          ++ show k
          ++ ")"
      }}}}
    Just i ->
      error
      $ "matmul: input operand "
      ++ show i
      ++ " does not have enough dimensions (has 0)"
  }

-- | An infix synonym for @matmul@.
(@) :: (HasDtype t, Floating t) => Tensor t -> Tensor t -> Tensor t
(@) = matmul

infixl 7 @

-- | Perform elementwise operation without cheking
--   for validity of arguments.
--
--   This function is not safe to use,
--   consider using @elementwise@ instead.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
unsafeElementwise :: (HasDtype a, HasDtype b, HasDtype c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
unsafeElementwise f x1 x2 =
  case copy x1 of {(Tensor shape stride offset dat1) ->
  case copy x2 of {(Tensor _ _ _ dat2) ->
  case V.zipWith f dat1 dat2 of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}}

-- | Perform elementwise operation with broadcasting.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
elementwise :: (HasDtype a, HasDtype b, HasDtype c) =>
  (a -> b -> c) -> Tensor a -> Tensor b -> Tensor c
elementwise f x1 x2 =
  case broadcast x1 x2 of {(x1, x2) ->
    if numel x1 == 1 then
      Data.Tensor.Functional.map (f $ item x1) x2
    else if numel x2 == 1 then
      Data.Tensor.Functional.map (flip f $ item x2) x1
    else unsafeElementwise f x1 x2
  }


-- Indexing
-- --------

-- | Get element of a tensor at a specified index.
--
--   Negative index values are supported.
(!) :: HasDtype t => Tensor t -> Index -> t
(!) x@(Tensor shape stride offset dat) index =
  case parseIndex shape $ V.fromList index of {index ->
  case tensorDtype x of {dtype ->
  case V.unsafeCast dat of {dataCChar ->
    unsafeDupablePerformIO
    $ [CU.exp| char * {
        get_elem(
          $vec-len:shape,
          $vec-ptr:(long long *stride),
          $(size_t offset),
          $(int dtype),
          $vec-ptr:(char *dataCChar),
          $vec-ptr:(size_t *index)
        )
      } |] >>= peek . castPtr
  }}}

-- | Return the value of a tensor with one element.
item :: (HasDtype t) => Tensor t -> t
item x@(Tensor shape _ _ _)
  | totalElems_ shape == 1 =
      x ! replicate (V.length shape) 0
  | otherwise =
    error
    $ "cannot get item from tensor of shape "
    ++ show shape


-- Unary operations
-- ----------------

dim :: (HasDtype t) => Tensor t -> Int
dim (Tensor shape _ _ _) = V.length shape

shape :: (HasDtype t) => Tensor t -> Index
shape (Tensor shape _ _ _) = shapeToIndex shape

-- | Total number of elements in a tensor.
numel :: (HasDtype t) => Tensor t -> Int
numel (Tensor shape _ _ _) = fromIntegral $ totalElems_ shape

-- | Total number of elements in a tensor.
numel_ :: (HasDtype t) => Tensor t -> CSize
numel_ (Tensor shape _ _ _) = totalElems_ shape

-- | Minimum value of a tensor.
min :: (HasDtype t, Ord t) => Tensor t -> t
min x@(Tensor shape stride offset dat)
  | totalElems_ shape /= 0 =
    case V.unsafeCast dat of {dataCChar ->
    case tensorDtype x of {dtype ->
      unsafePerformIO
      $ alloca (
        \ outCCharPtr -> do
          [CU.exp| void {
            tensor_min(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $(char *outCCharPtr)
            )
          } |]
          case castPtr outCCharPtr of {outPtr ->
            peek outPtr
          }
      )
    }}
  | otherwise =
    error "zero-size array to reduction operation minimum which has no identity"

-- | Maximum value of a tensor.
max :: (HasDtype t, Ord t) => Tensor t -> t
max x@(Tensor shape stride offset dat)
  | totalElems_ shape /= 0 =
    case V.unsafeCast dat of {dataCChar ->
    case tensorDtype x of {dtype ->
      unsafePerformIO
      $ alloca (
        \ outCCharPtr -> do
          [CU.exp| void {
            tensor_max(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(int dtype),
              $vec-ptr:(char *dataCChar),
              $(char *outCCharPtr)
            )
          } |]
          case castPtr outCCharPtr of {outPtr ->
            peek outPtr
          }
      )
    }}
  | otherwise =
    error "zero-size array to reduction operation maximum which has no identity"

-- | Sum elements of a tensor.
sum :: (HasDtype t, Num t) => Tensor t -> t
sum x@(Tensor shape stride offset dat) =
  case V.unsafeCast dat of {dataCChar ->
  case tensorDtype x of {dtype ->
    unsafePerformIO
    $ alloca (
      \ outCCharPtr -> do
        [CU.exp| void {
          tensor_sum(
            $vec-len:shape,
            $vec-ptr:(size_t *shape),
            $vec-ptr:(long long *stride),
            $(size_t offset),
            $(int dtype),
            $vec-ptr:(char *dataCChar),
            $(char *outCCharPtr)
          )
        } |]
        case castPtr outCCharPtr of {outPtr ->
          peek outPtr
        }
    )
  }}

-- | Take a mean of elements in a tensor.
mean :: (HasDtype t, Fractional t) => Tensor t -> t
mean x = Data.Tensor.Functional.sum x / fromIntegral (numel_ x)

-- | Sum elements of a tensor along specified dims.
--
--   Negative dim values are supported.
--
--   Signature: @tensor -> dims -> keepDims -> tensor@
sumAlongDims :: (HasDtype t, Num t) => Tensor t -> [Int] -> Bool -> Tensor t
sumAlongDims x@(Tensor shape stride offset dat) dims keepDims =
  case V.length shape of {nDims ->
  case Prelude.map (normalizeItem nDims) dims of {normDims ->
    if all (\ dim -> 0 <= dim && dim < nDims) normDims then
      if length (nub normDims) == length normDims then
        case filter (not . flip elem normDims) [0 .. nDims - 1] ++ normDims of {dims ->
        case (sortDims shape dims, sortDims stride dims) of {(sortedShape, sortedStride) ->
        case fromIntegral $ nDims - length normDims of {startSumDim ->
        case (
          if keepDims then
            V.zipWith (
              \ axis dim ->
                if axis `elem` normDims then
                  1
                else dim
            ) (V.fromList [0 .. nDims - 1]) shape
          else V.take (fromIntegral startSumDim) sortedShape
        ) of {newShape ->
        case tensorDtype x of {dtype ->
        case sizeOfElem dat of {elemSize ->
        case V.unsafeCast dat of {dataCChar ->
          Tensor newShape (computeStride elemSize newShape) 0
          $ unsafePerformIO
          $ do
            mutableData <- VM.new $ fromIntegral $ totalElems_ newShape
            case VM.unsafeCast mutableData of {mutableDataCChar ->
              [CU.exp| void {
                sum_along_dims(
                  $(int startSumDim),
                  $vec-len:shape,
                  $vec-ptr:(size_t *sortedShape),
                  $vec-ptr:(long long *sortedStride),
                  $(size_t offset),
                  $(int dtype),
                  $(size_t elemSize),
                  $vec-ptr:(char *dataCChar),
                  $vec-ptr:(char *mutableDataCChar)
                )
              } |]
            }
            V.unsafeFreeze mutableData
        }}}}}}}
      else
        error
        $ "duplicate elements in dims "
        ++ show dims
    else
      error
      $ "dims "
      ++ show dims
      ++ " are out of bounds for "
      ++ show nDims
      ++ "-dimensional tensor"
  }}

-- | Sum elements of a tensor along specified dim.
--
--   Negative dim values are supported.
--
--   Signature: @tensor -> dim -> tensor@
sumAlongDim :: (HasDtype t, Num t) => Tensor t -> Int -> Tensor t
sumAlongDim x dim = sumAlongDims x [dim] False

-- | ReLU activation function @map (max 0)@.
relu :: (HasDtype t, Ord t) => Tensor t -> Tensor t
relu x@(Tensor shape stride offset dat) =
    case tensorDtype x of {dtype ->
    case V.unsafeCast dat of {dataCChar ->
      Tensor shape (computeStride (sizeOfElem dat) shape) 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems_ shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            tensor_relu(
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

-- | Return a contiguous copy of a tensor.
copy :: (HasDtype t) => Tensor t -> Tensor t
copy x@(Tensor shape stride offset dat) =
  case sizeOfElem dat of {elemSize ->
  case V.unsafeCast dat of {dataCChar ->
  case computeStride elemSize shape of {contiguousStride ->
    -- If all tensor elements point to the same value
    if V.and $ V.zipWith (
      \ dim stride -> dim == 1 || stride == 0
    ) shape stride then
      Tensor shape contiguousStride 0
      $ V.replicate (fromIntegral $ totalElems_ shape)
      $ unsafeDupablePerformIO
      $ V.unsafeWith dataCChar
      $ \ dataCCharPtr ->
        peek
        $ castPtr
        $ advancePtr dataCCharPtr
        $ fromIntegral offset
    -- If tensor is not contiguous, deep copy
    else if contiguousStride /= stride then
      Tensor shape contiguousStride 0
      $ unsafePerformIO
      $ do
        mutableData <- VM.new $ fromIntegral $ totalElems_ shape
        case VM.unsafeCast mutableData of {mutableDataCChar ->
          [CU.exp| void {
            copy(
              $vec-len:shape,
              $vec-ptr:(size_t *shape),
              $vec-ptr:(long long *stride),
              $(size_t offset),
              $(size_t elemSize),
              $vec-ptr:(char *dataCChar),
              $vec-ptr:(char *mutableDataCChar)
            )
          } |]
        }
        V.unsafeFreeze mutableData
    -- If tensor is already contiguous, take a view
    else
      Tensor shape stride 0
        $ V.unsafeCast
        $ V.slice (fromIntegral offset)
          (fromIntegral $ totalElems_ shape * elemSize) dataCChar
  }}}

-- | Convert a tensor to a vector.
toVector :: (HasDtype t) => Tensor t -> Vector t
toVector = tensorData . copy

-- | Cast a tensor to a different dtype.
astype :: forall a b. (HasDtype a, HasDtype b) => Tensor a -> Tensor b
astype x@(Tensor shape stride offset dat) =
  case empty :: Tensor b of {sampleTensor ->
  case tensorDtype sampleTensor of {dtype_to ->
  case tensorDtype x of {dtype_from ->
    if dtype_from == dtype_to then
      Tensor shape stride offset $ V.unsafeCast dat
    else
      case V.unsafeCast dat of {dataCChar ->
        Tensor shape (computeStride (sizeOfElem $ tensorData sampleTensor) shape) 0
        $ unsafePerformIO
        $ do
          mutableData <- VM.new $ fromIntegral $ totalElems_ shape
          case VM.unsafeCast mutableData of {mutableDataCChar ->
            [CU.exp| void {
              tensor_astype(
                $vec-len:shape,
                $vec-ptr:(size_t *shape),
                $vec-ptr:(long long *stride),
                $(size_t offset),
                $(int dtype_from),
                $vec-ptr:(char *dataCChar),
                $(int dtype_to),
                $vec-ptr:(char *mutableDataCChar)
              )
            } |]
          }
          V.unsafeFreeze mutableData
      }
  }}}

-- | Return a tensor with last 2 axes transposed.
transpose :: HasDtype t => Tensor t -> Tensor t
transpose x = swapDims x (-1, -2)

-- | Return a copy of the tensor collapsed into one dimension.
flatten :: (HasDtype t) => Tensor t -> Tensor t
flatten x =
  case copy x of {(Tensor shape stride offset dat) ->
    Tensor (V.singleton $ totalElems_ shape)
      (V.singleton $ fromIntegral $ sizeOfElem dat) offset dat
  }

-- | Give a new shape to a tensor without changing its data.
--
--   No more than one dim is allowed to be (-1), in which case
--   it is inferred from the current shape.
--
--   Signature: @tensor -> newShape -> tensor@
view :: (HasDtype t) => Tensor t -> Index -> Tensor t
view x@(Tensor shape stride offset dat) newShape =
  case parseNewShape shape newShape of {parsedNewShape ->
    if totalElems_ shape == totalElems_ parsedNewShape then
      case sizeOfElem dat of {elemSize ->
      case computeStride elemSize shape of {contiguousStride ->
      case computeStride elemSize parsedNewShape of {newStride ->
        if stride == contiguousStride then
          Tensor parsedNewShape newStride offset dat
        else
          case copy x of {(Tensor _ _ offset dat) ->
            Tensor parsedNewShape newStride offset dat
          }
      }}}
    else
      error
      $ "cannot reshape tensor of shape "
      ++ show shape
      ++ " into shape "
      ++ show newShape
  }

-- | Insert a new dim into a tensor.
--
--   Negative dim values are supported.
--
--   Signature: @tensor -> dim -> tensor@
insertDim :: (HasDtype t) => Tensor t -> Int -> Tensor t
insertDim (Tensor shape stride offset dat) dim =
  case V.length shape of {nDims ->
  case normalizeNewDim nDims dim of {normDim ->
    if 0 <= normDim && normDim <= nDims then
      Tensor
      (V.concat [
          V.take normDim shape,
          V.singleton 1,
          V.drop normDim shape])
      (V.concat [
          V.take normDim stride,
          V.singleton 0,
          V.drop normDim stride])
      offset dat
    else
      error
      $ "cannot insert dim "
      ++ show dim
      ++ " in tensor of shape "
      ++ show shape
  }}

-- | Sequentially insert new dims into a tensor.
--
--   Negative dim values are supported.
--
--   Signature: @tensor -> dims -> tensor@
insertDims :: (HasDtype t) => Tensor t -> [Int] -> Tensor t
insertDims = Data.List.foldl' insertDim

-- -- | tensor -> dim -> times
-- repeatAlongDim :: (HasDtype t) => Tensor t -> Int -> Int -> Tensor t
-- repeatAlongDim x@(Tensor shape _) dim times =
--   tensor newShape (
--     \ index ->
--         x !? (index // [(nDim, (index ! nDim) `mod` currentDim)])
--   )
--   where
--     nDim = normalizeItem (V.length shape) dim
--     currentDim = shape ! nDim
--     newShape = shape // [(nDim, currentDim * times)]
--     -- debugPrint x =
--     --   unsafePerformIO $ print x >> return x

-- | Interchange two dims of a tensor.
--
--   Negative dim values are supported.
--
--   Signature: @tensor -> (dim1, dim2) -> tensor@
swapDims :: (HasDtype t) =>
  Tensor t -> (Int, Int) -> Tensor t
swapDims x@(Tensor shape stride offset dat) (dim1, dim2) =
  case V.length shape of {nDims ->
  case normalizeItem nDims dim1 of {normDim1 ->
  case normalizeItem nDims dim2 of {normDim2 ->
    if 0 <= normDim1 && normDim1 < nDims &&
       0 <= normDim2 && normDim2 < nDims then
      Tensor
      (swapElementsAt shape (normDim1, normDim2))
      (swapElementsAt stride (normDim1, normDim2))
      offset dat
    else
      error
      $ "cannot swap dims "
      ++ show dim1
      ++ ", "
      ++ show dim2
      ++ " of tensor of shape "
      ++ show shape
  }}}

-- | Map a function over a tensor.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
map :: (HasDtype a, HasDtype b) => (a -> b) -> Tensor a -> Tensor b
map f x =
  case copy x of {(Tensor shape stride offset dat) ->
  case V.map f dat of {dat ->
    Tensor shape (computeStride (sizeOfElem dat) shape) 0 dat
  }}

-- | Left fold with strict accumulator.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
foldl' :: HasDtype b => (a -> b -> a) -> a -> Tensor b -> a
foldl' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldl' f accum dat
  }

-- | Right fold with strict accumulator.
--
--   /WARNING:/ This function involves copying and
--   can be less efficient than native tensor operations.
--   Consider using them if possible.
foldr' :: HasDtype b => (b -> a -> a) -> a -> Tensor b -> a
foldr' f accum x =
  case copy x of {(Tensor shape stride offset dat) ->
    V.foldr' f accum dat
  }

{-# INLINE tensor_ #-}
{-# INLINE full_ #-}
{-# INLINE randRange #-}
{-# INLINE randRangeM #-}
{-# INLINE randn #-}
{-# INLINE randnM #-}
{-# INLINE eye #-}

{-# INLINE broadcast #-}
{-# INLINE (//) #-}
{-# INLINE (%) #-}
{-# INLINE matmul #-}
{-# INLINE unsafeElementwise #-}

{-# INLINE (!) #-}

{-# INLINE copy #-}
{-# INLINE min #-}
{-# INLINE max #-}
{-# INLINE sum #-}
{-# INLINE sumAlongDims #-}
{-# INLINE map #-}
{-# INLINE foldl' #-}
{-# INLINE foldr' #-}
