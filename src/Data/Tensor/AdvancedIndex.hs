{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE FlexibleInstances #-}

module Data.Tensor.AdvancedIndex where

import Control.Monad
import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List
import Data.Either
import Data.Tensor.PlainIndex
import Data.Tensor.Size
import Data.Tensor.Definitions
import Data.Tensor.Functional (insertDims, broadcastN, broadcastTo, dim)
import qualified Data.Tensor.Functional as T
import Data.Tensor.Instances
import qualified Data.Tensor.Boolean

import System.IO.Unsafe
import Foreign
import Foreign.C.Types
import qualified Language.C.Inline as C
import qualified Language.C.Inline.Unsafe as CU

-- Use vector anti-quoters.
C.context (C.baseCtx <> C.vecCtx)
-- Include C utils.
C.include "cbits/core/core.h"


deriving instance Eq Indexer

instance Show Indexer where
  show (I i) = show i
  show A = ":"
  show (S start) = show start ++ ":"
  show (E end) = ':' : show end
  show (index :. step) = show index ++ ":" ++ show step
  show (T tI) = show tI
  show None = "None"
  show Ell = "..."

instance {-# OVERLAPPING #-} Show Indexers where
  show indexers = "[" ++ intercalate ", " (map show indexers) ++ "]"

instance Num Indexer where
  (+) (I a) (I b) = I $ a + b
  (+) _ _ = error "illegal operation for indexers"
  (-) (I a) (I b) = I $ a - b
  (-) _ _ = error "illegal operation for indexers"
  (*) (I a) (I b) = I $ a * b
  (*) _ _ = error "illegal operation for indexers"
  abs (I i) = I $ abs i
  abs _ = error "illegal operation for indexers"
  signum (I i) = I $ signum i
  signum _ = error "illegal operation for indexers"
  fromInteger = I . fromInteger

instance Ord Indexer where
  compare (I a) (I b) = compare a b
  compare _ _ = error "illegal operation for indexers"
  (<) (I a) (I b) = a < b
  (<) _ _ = error "illegal operation for indexers"
  (<=) (I a) (I b) = a <= b
  (<=) _ _ = error "illegal operation for indexers"
  (>) (I a) (I b) = a > b
  (>) _ _ = error "illegal operation for indexers"
  (>=) (I a) (I b) = a >= b
  (>=) _ _ = error "illegal operation for indexers"
  max (I a) (I b) = I $ max a b
  max _ _ = error "illegal operation for indexers"
  min (I a) (I b) = I $ min a b
  min _ _ = error "illegal operation for indexers"

instance Real Indexer where
  toRational (I i) = toRational i
  toRational _ = error "illegal operation for indexers"

instance Enum Indexer where
  toEnum = I . toEnum
  fromEnum (I i) = fromEnum i
  fromEnum _ = error "illegal operation for indexers"

instance Integral Indexer where
  quot (I a) (I b) = I (quot a b)
  quot _ _ = error "illegal operation for indexers"
  rem (I a) (I b) = I (rem a b)
  rem _ _ = error "illegal operation for indexers"
  div (I a) (I b) = I (div a b)
  div _ _ = error "illegal operation for indexers"
  mod (I a) (I b) = I (mod a b)
  mod _ _ = error "illegal operation for indexers"
  quotRem (I a) (I b) =
    case quotRem a b of {(a, b) ->
      (I a, I b)
    }
  quotRem _ _ = error "illegal operation for indexers"
  divMod (I a) (I b) =
    case divMod a b of {(a, b) ->
      (I a, I b)
    }
  divMod _ _ = error "illegal operation for indexers"
  toInteger (I i) = toInteger i
  toInteger _ = error "illegal operation for indexers"

-- | Parse @None@ in indexers.
--
--   Signature: @nDims -> indexers -> (dimsToInsert, parsedIndexers)@
parseNoneIndexer :: Int -> Indexers -> ([Int], Indexers)
parseNoneIndexer nDims indexers =
  case map (
    \case
      None    -> A
      indexer -> indexer
  ) indexers of {parsedIndexers ->
    case elemIndices Ell indexers of
      [] ->
        (elemIndices None indexers, parsedIndexers)
      [dim] ->
        (elemIndices None (take dim indexers)
         ++ map (-1 -) (
          elemIndices None
          $ reverse
          $ drop dim indexers
        ), parsedIndexers)
      _ ->
        error "indexers can only have a single ellipsis \'Ell\'"
  }

-- | Parse @Ell@ and pad indexers to @nDims@ length.
--
--   Signature: @nDims -> indexers -> parsedIndexers@
parseEllIndexer :: Int -> Indexers -> Indexers
parseEllIndexer nDims indexers =
  case elemIndices Ell indexers of
    [] ->
      indexers ++ replicate (nDims - length indexers) A
    [dim] ->
      take dim indexers
      ++ replicate (nDims - length indexers + 1) A
      ++ drop (dim + 1) indexers
    _ ->
      error "indexers can only have a single ellipsis \'Ell\'"

-- | Parse integer tensor indexers and validate their correctness.
--
--   Signature: @shape -> indexers -> (tensorIndices, indexers)@
parseTensorIndexer :: Shape -> Indexers -> ([Either (Int, Tensor CLLong) Int], Indexers)
parseTensorIndexer shape indexers =
  (zipWith (
    \ axis (dim, indexer) ->
      case indexer of
        T tI@(Tensor shape stride offset dat) ->
          case V.unsafeCast dat of {dataCChar ->
            unsafePerformIO
            $ alloca
            $ \ outPtr -> do
                isValid <- [CU.exp| int {
                    validate_tensor_index(
                      $(size_t dim),
                      $vec-len:shape,
                      $vec-ptr:(size_t *shape),
                      $vec-ptr:(long long *stride),
                      $(size_t offset),
                      $vec-ptr:(char *dataCChar),
                      $(long long *outPtr)
                    )
                  } |]
                if toBool isValid then
                  return $ Left (axis, tI)
                else do
                  i <- peek outPtr
                  error
                    $ "index "
                    ++ show i
                    ++ " is out of bounds for dim "
                    ++ show axis
                    ++ " with size "
                    ++ show dim
          }
        _ ->
          Right axis
  ) [0 .. V.length shape - 1]
  $ filter (
    \case
      (_, I _) -> False
      _        -> True
  )
  $ zip (V.toList shape) indexers,
  map (
    \case
      T _     -> A
      indexer -> indexer
  ) indexers)

-- | Parse negative indexer values.
--
--   Signature: @i -> dim -> indexer -> step -> normIndexer@
normalizeSliceIndexer :: Int -> Int -> Indexer -> Int -> Indexer
normalizeSliceIndexer axis dim (I i) step =
  case normalizeItem dim i of {normI ->
    if 0 <= normI && normI < dim then
      I normI
    else
      error
      $ "index "
      ++ show i
      ++ " is out of bounds for dim "
      ++ show axis
      ++ " with size "
      ++ show dim
  }
normalizeSliceIndexer axis dim indexer step
  | step > 0 =
    case indexer of
      A -> 0 :. dim :. step
      S start ->
        case min dim $ max 0 $ normalizeItem dim start of {start ->
          I start :. dim :. step
        }
      E end ->
        case min dim $ max 0 $ normalizeItem dim end of {end ->
          0 :. end :. step
        }
      I start :. end ->
        case min dim $ max 0 $ normalizeItem dim start of {start ->
        case min dim $ max 0 $ normalizeItem dim end of {end ->
          I start :. max start end :. step
        }}
      badIndexer ->
        error $ "incorrect indexer " ++ show (badIndexer :. step)
  | step < 0 =
    case indexer of
      A -> I dim - 1 :. -1 :. step
      S start ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim start of {start ->
          I start :. -1 :. step
        }
      E end ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim end of {end ->
          I dim - 1 :. end :. step
        }
      I start :. end ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim start of {start ->
        case min (dim - 1) $ max (-1) $ normalizeItem dim end of {end ->
          I (max start end) :. end :. step
        }}
      badIndexer ->
        error $ "incorrect indexer " ++ show (badIndexer :. step)
  | otherwise =
    error "indexer step cannot be zero"

-- | Parse indexers and validate their correctness.
--
--   Signature: @shape -> indexers -> parsedIndexers@
parseSliceIndexers :: Shape -> Indexers -> Indexers
parseSliceIndexers shape indexers =
  case V.length shape of {nDims ->
    if length indexers == nDims then
      zipWith3 (
        \ axis dim indexer ->
          case indexer of
            I _ :. step ->
              normalizeSliceIndexer axis (fromIntegral dim) indexer 1
            indexer :. step ->
              normalizeSliceIndexer axis (fromIntegral dim) indexer step
            _ ->
              normalizeSliceIndexer axis (fromIntegral dim) indexer 1
      ) [0 .. nDims - 1]
      (V.toList shape) indexers
    else
      error
      $ "too many indices for tensor: tensor is "
      ++ show nDims
      ++ "-dimensional, but "
      ++ show (length indexers)
      ++ " were indexed"
  }

-- | Perform slicing.
slice :: (Shape, Stride, CSize) -> Indexers -> (Shape, Stride, CSize)
slice (shape, stride, offset) indexers =
  case parseSliceIndexers shape indexers of {indexers ->
    (V.fromList $ map (
      \ (I start :. end :. step) -> fromIntegral $ (end - start) `div` step
    ) $ filter (
      \case
        I _ -> False
        _   -> True
    ) indexers,
    V.fromList $ map (
      \ (_ :. _ :. step, stride) ->
        stride * fromIntegral step
    ) $ filter (
      \ (indexer, stride) ->
        case indexer of
          I _ -> False
          _   -> True
    ) $ zip indexers $ V.toList stride,
    (+) offset $ sum $ zipWith (
      \ indexer stride ->
        case indexer of
          I start :. end :. step ->
            fromIntegral $ fromIntegral start * stride
          I i ->
            fromIntegral $ fromIntegral i * stride
    ) indexers $ V.toList stride)
  }

tensorIndex :: (HasDtype t) => Tensor t -> ([(Int, Tensor CLLong)], [Int]) -> Tensor t
tensorIndex x@(Tensor shape stride offset dat) (tensorIndices, residualDims) =
  case map fst tensorIndices of {tensorDims ->
  case broadcastN $ map snd tensorIndices of {tensorIndices ->
  case fromIntegral $ length tensorIndices of {nIndices ->
  case fromIntegral $ dim $ head tensorIndices of {indexNDims ->
  case tensorShape $ head tensorIndices of {indexShape ->
  -- If tensor indexers are separated by None, Ell or Slice,
  -- indexed dims come first in the resulting tensor.
  case (
    if findGap tensorDims then
      0
    else fromIntegral $ head tensorDims
  ) of {startIndexDim ->
  case (
    case tensorDims ++ residualDims of {dims ->
      (sortDims shape dims, sortDims stride dims)
    }
  ) of {(shape, stride) ->
  -- unsafePerformIO $ print (tensorDims, residualDims, shape, stride) >> return (
  case V.concat [
    V.take (fromIntegral startIndexDim) $ V.drop (fromIntegral nIndices) shape,
    indexShape,
    V.drop (fromIntegral $ startIndexDim + nIndices) shape
  ] of {newShape ->
  case sizeOfElem dat of {elemSize ->
  case computeStride elemSize newShape of {contiguousStride ->
  case V.concat [
    V.take (fromIntegral nIndices) $ V.drop (fromIntegral startIndexDim) contiguousStride,
    V.take (fromIntegral startIndexDim) contiguousStride,
    V.drop (fromIntegral $ startIndexDim + nIndices) contiguousStride
  ] of {newStride ->
  case V.unsafeCast dat of {dataCChar ->
      Tensor newShape contiguousStride 0
      $ unsafePerformIO
      $ allocaBytes (fromIntegral nIndices * sizeOf (undefined :: Ptr CLLong))
      $ \ indexStridesPtr ->
        allocaBytes (fromIntegral nIndices * sizeOf (undefined :: CSize))
        $ \ indexOffsetsPtr ->
          allocaBytes (fromIntegral nIndices * sizeOf (undefined :: Ptr CChar))
          $ \ indexDatPtr -> do
            zipWithM_ (
              \ i (Tensor _ stride offset dat) -> do
                V.unsafeWith stride $ poke $ advancePtr indexStridesPtr i
                poke (advancePtr indexOffsetsPtr i) offset
                V.unsafeWith (V.unsafeCast dat) $ poke $ advancePtr indexDatPtr i
              ) [0 .. fromIntegral nIndices - 1] tensorIndices
            mutableData <- VM.new $ fromIntegral $ totalElems_ newShape
            case VM.unsafeCast mutableData of {mutableDataCChar ->
              [CU.exp| void {
                tensor_index(
                  $(int nIndices),
                  $(int indexNDims),
                  $vec-ptr:(size_t *indexShape),
                  $(long long **indexStridesPtr),
                  $(size_t *indexOffsetsPtr),
                  $(char **indexDatPtr),
                  $vec-len:shape,
                  $vec-ptr:(size_t *shape),
                  $(size_t elemSize),
                  $vec-ptr:(long long *stride),
                  $(size_t offset),
                  $vec-ptr:(char *dataCChar),
                  $vec-ptr:(long long *newStride),
                  0,
                  $vec-ptr:(char *mutableDataCChar)
                )
              } |]
            }
            V.unsafeFreeze mutableData
  }}}}}}}}}}}}

-- | Perform slicing and advanced indexing.
--
--   Negative values in indexer bounds and index tensors are supported.
(!:) :: (HasDtype t) => Tensor t -> Indexers -> Tensor t
(!:) x@(Tensor shape _ _ _) indexers =
  case parseNoneIndexer (V.length shape) indexers of {(dimsToInsert, indexers) ->
  case insertDims x dimsToInsert of {x@(Tensor shape stride offset dat) ->
  case parseEllIndexer (V.length shape) indexers of {indexers ->
  case parseTensorIndexer shape indexers of {(tensorIndices, indexers) ->
  case slice (shape, stride, offset) indexers of {(shape, stride, offset) ->
  case Tensor shape stride offset dat of {x ->
  case partitionEithers tensorIndices of {(tensorIndices, residualDims) ->
    if null tensorIndices then
      x
    else tensorIndex x (tensorIndices, residualDims)
  }}}}}}}

-- | Assign a subtensor using slicing and advanced indexing.
--
--   Negative values in indexer bounds and index tensors are supported.
--
--   Signature: @tensor -> (indexers, subtensor) -> updatedTensor@
(!=) :: (HasDtype t) => Tensor t -> (Indexers, Tensor t) -> Tensor t
(!=) xTo@(Tensor shapeTo strideTo offsetTo dat) (indexers, xFrom) =
  case sizeOfElem dat of {elemSize ->
  case computeStride elemSize shapeTo of {contiguousStride ->
  case parseNoneIndexer (V.length shapeTo) indexers of {(dimsToInsert, indexers) ->
  case insertDims (Tensor shapeTo contiguousStride 0 dat) dimsToInsert
  of {(Tensor shape stride offset dat) ->
  case parseEllIndexer (V.length shape) indexers of {indexers ->
  case parseTensorIndexer shape indexers of {(tensorIndices, indexers) ->
  case slice (shape, stride, offset) indexers of {(shape, stride, offset) ->
  case partitionEithers tensorIndices of {(tensorIndices, residualDims) ->
  case map fst tensorIndices of {tensorDims ->
  case broadcastN $ map snd tensorIndices of {tensorIndices ->
  case fromIntegral $ length tensorIndices of {nIndices ->
  -- If tensor indexers are separated by None, Ell or Slice,
  -- indexed dims come first in the resulting tensor.
  case (
    if not $ null tensorIndices then
      (fromIntegral $ dim $ head tensorIndices,
      tensorShape $ head tensorIndices,
      if findGap tensorDims then
        0
      else fromIntegral $ head tensorDims)
    else (0, V.empty, 0)
  ) of {(indexNDims, indexShape, startIndexDim) ->
  case (
    case tensorDims ++ residualDims of {dims ->
      (sortDims shape dims, sortDims stride dims)
    }
  ) of {(shape, stride) ->
  case V.concat [
    V.take (fromIntegral startIndexDim) $ V.drop (fromIntegral nIndices) shape,
    indexShape,
    V.drop (fromIntegral $ startIndexDim + nIndices) shape
  ] of {newShape ->
  -- unsafePerformIO $ print (tensorDims, residualDims, shape, stride, newShape) >> return (
  case broadcastTo xFrom newShape of {(Tensor _ strideFrom offsetFrom datFrom) ->
  case V.unsafeCast dat of {dataCChar ->
  case V.unsafeCast datFrom of {dataFromCChar ->
      Tensor shapeTo contiguousStride 0
      $ unsafePerformIO
      $ allocaBytes (fromIntegral nIndices * sizeOf (undefined :: Ptr CLLong))
      $ \ indexStridesPtr ->
        allocaBytes (fromIntegral nIndices * sizeOf (undefined :: CSize))
        $ \ indexOffsetsPtr ->
          allocaBytes (fromIntegral nIndices * sizeOf (undefined :: Ptr CChar))
          $ \ indexDatPtr -> do
            zipWithM_ (
              \ i (Tensor _ stride offset dat) -> do
                V.unsafeWith stride $ poke $ advancePtr indexStridesPtr i
                poke (advancePtr indexOffsetsPtr i) offset
                V.unsafeWith (V.unsafeCast dat) $ poke $ advancePtr indexDatPtr i
              ) [0 .. fromIntegral nIndices - 1] tensorIndices
            mutableData <- VM.new $ fromIntegral $ totalElems_ shapeTo
            case VM.unsafeCast mutableData of {mutableDataCChar -> do
              [CU.exp| void {
                copy(
                  $vec-len:shapeTo,
                  $vec-ptr:(size_t *shapeTo),
                  $vec-ptr:(long long *strideTo),
                  $(size_t offsetTo),
                  $(size_t elemSize),
                  $vec-ptr:(char *dataCChar),
                  $vec-ptr:(char *mutableDataCChar)
                )
              } |]
              [CU.exp| void {
                tensor_index_assign(
                  $(int nIndices),
                  $(int indexNDims),
                  $vec-ptr:(size_t *indexShape),
                  $(long long **indexStridesPtr),
                  $(size_t *indexOffsetsPtr),
                  $(char **indexDatPtr),
                  $vec-len:shape,
                  $vec-ptr:(size_t *shape),
                  $(size_t elemSize),
                  $vec-ptr:(long long *strideFrom),
                  $(size_t offsetFrom),
                  $vec-ptr:(char *dataFromCChar),
                  $vec-ptr:(long long *stride),
                  $(size_t offset),
                  $vec-ptr:(char *mutableDataCChar)
                )
              } |]
            }
            V.unsafeFreeze mutableData
  }}}}}}}}}}}}}}}}}

{-# INLINE parseNoneIndexer #-}
{-# INLINE parseEllIndexer #-}
{-# INLINE parseTensorIndexer #-}
{-# INLINE normalizeSliceIndexer #-}
{-# INLINE parseSliceIndexers #-}
{-# INLINE slice #-}
{-# INLINE tensorIndex #-}
{-# INLINE (!:) #-}
{-# INLINE (!=) #-}
