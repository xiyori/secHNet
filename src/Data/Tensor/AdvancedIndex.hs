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
import Data.Tensor.Functional (insertDims, broadcastN, dim)
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
  show indexers = "[" ++ go indexers ++ "]"
    where
      go [i] = show i
      go (i:is) = show i ++ ", " ++ go is

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
--   Signature: @shape -> indexers -> (tensorIndexers, sliceIndexers)@
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
normalizeIndexer :: Int -> Int -> Indexer -> Int -> Indexer
normalizeIndexer axis dim (I i) step =
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
normalizeIndexer axis dim indexer step
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
parseIndexers :: Shape -> Indexers -> Indexers
parseIndexers shape indexers =
  case V.length shape of {nDims ->
    if length indexers == nDims then
      zipWith3 (
        \ axis dim indexer ->
          case indexer of
            I _ :. step ->
              normalizeIndexer axis (fromIntegral dim) indexer 1
            indexer :. step ->
              normalizeIndexer axis (fromIntegral dim) indexer step
            _ ->
              normalizeIndexer axis (fromIntegral dim) indexer 1
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

-- | Perform slicing only.
slice :: (HasDtype t) => Tensor t -> Indexers -> Tensor t
slice x@(Tensor shape stride offset dat) indexers =
  case parseIndexers shape indexers of {indexers ->
    Tensor (V.fromList $ map (
      \ (I start :. end :. step) -> fromIntegral $ (end - start) `div` step
    ) $ filter (
      \case
        I _ -> False
        _   -> True
    ) indexers)
    (V.fromList $ map (
      \ (_ :. _ :. step, stride) ->
        stride * fromIntegral step
    ) $ filter (
      \ (indexer, stride) ->
        case indexer of
          I _ -> False
          _   -> True
    ) $ zip indexers $ V.toList stride)
    ((+) offset $ sum $ zipWith (
      \ indexer stride ->
        case indexer of
          I start :. end :. step ->
            fromIntegral $ fromIntegral start * stride
          I i ->
            fromIntegral $ fromIntegral i * stride
    ) indexers $ V.toList stride) dat
  }

-- | Perform slicing and advanced indexing.
--
--   Negative values in indexer bounds and index tensors are supported.
(!:) :: (HasDtype t) => Tensor t -> Indexers -> Tensor t
(!:) x@(Tensor shape _ _ _) indexers =
  case parseNoneIndexer (V.length shape) indexers of {(dimsToInsert, indexers) ->
  case insertDims x dimsToInsert of {x@(Tensor shape _ _ _) ->
  case parseEllIndexer (V.length shape) indexers of {indexers ->
  case parseTensorIndexer shape indexers of {(tensorIndexers, indexers) ->
  case slice x indexers of {x@(Tensor shape stride offset dat) ->
  case partitionEithers tensorIndexers of {(tensorIndexers, residualDims) ->
    if not $ null tensorIndexers then
      case map fst tensorIndexers of {tensorDims ->
      -- If tensor indexers are separated by None, Ell or Slice,
      -- indexed dims come first in the resulting tensor.
      case (
        if findGap tensorDims then
          case tensorDims ++ residualDims of {dims ->
            (0, sortDims shape dims, sortDims stride dims)
          }
        else (fromIntegral $ head tensorDims, shape, stride)
      ) of {(startIndexDim, shape, stride) ->
      case broadcastN $ map snd tensorIndexers of {tensorIndexers ->
      case fromIntegral $ length tensorIndexers of {nIndices ->
      case fromIntegral $ dim $ head tensorIndexers of {indexNDims ->
      case tensorShape $ head tensorIndexers of {indexShape ->
      case V.concat [
        V.take (fromIntegral startIndexDim) shape,
        indexShape,
        V.drop (fromIntegral (startIndexDim + nIndices)) shape
      ] of {newShape ->
      -- unsafePerformIO
      -- $ print (startIndexDim, nIndices, indexNDims, indexShape, newShape) >> return (
      case sizeOfElem dat of {elemSize ->
      case V.unsafeCast dat of {dataCChar ->
          Tensor newShape (computeStride elemSize newShape) 0
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
                  ) [0 .. fromIntegral nIndices - 1] tensorIndexers
                mutableData <- VM.new $ fromIntegral $ totalElems_ newShape
                case VM.unsafeCast mutableData of {mutableDataCChar ->
                  [CU.exp| void {
                    tensor_index(
                      $vec-len:shape,
                      $vec-ptr:(size_t *shape),
                      $vec-ptr:(long long *stride),
                      $(size_t offset),
                      $(size_t elemSize),
                      $vec-ptr:(char *dataCChar),
                      $(int startIndexDim),
                      $(int nIndices),
                      $(int indexNDims),
                      $vec-ptr:(size_t *indexShape),
                      $(long long **indexStridesPtr),
                      $(size_t *indexOffsetsPtr),
                      $(char **indexDatPtr),
                      $vec-ptr:(char *mutableDataCChar)
                    )
                  } |]
                }
                V.unsafeFreeze mutableData
      }}}}}}}}}
    else x
  }}}}}}
    where
      findGap [] = False
      findGap [_] = False
      findGap (dim1 : dim2 : dims) = dim2 - dim1 > 1 || findGap (dim2 : dims)
      sortDims vec = V.map (vec V.!) . V.fromList

{-# INLINE parseNoneIndexer #-}
{-# INLINE parseEllIndexer #-}
{-# INLINE parseTensorIndexer #-}
{-# INLINE normalizeIndexer #-}
{-# INLINE parseIndexers #-}
{-# INLINE slice #-}
{-# INLINE (!:) #-}
