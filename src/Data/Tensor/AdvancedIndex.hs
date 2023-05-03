{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE FlexibleInstances #-}

module Data.Tensor.AdvancedIndex where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.List
import Data.Tensor.PlainIndex
import Data.Tensor.Size
import Data.Tensor.Definitions
import Data.Tensor.Functional (insertDims)
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
parseNone :: Int -> Indexers -> ([Int], Indexers)
parseNone nDims indexers =
  case filter (
    \case
      I _ -> False
      _   -> True
  ) indexers of {intervalsOnly ->
  case elemIndices Ell indexers of
    [] ->
      (elemIndices None intervalsOnly,
       filter (/= None) indexers)
    [dim] ->
      (elemIndices None (take dim intervalsOnly)
       ++ map (nDims - length intervalsOnly + dim +)
          (elemIndices None (drop dim intervalsOnly)),
       filter (/= None) indexers)
    _ ->
      error "indexers can only have a single ellipsis \'Ell\'"
  }

-- | Parse @Ell@ and pad indexers to @nDims@ length.
--
--   Signature: @nDims -> indexers -> parsedIndexers@
parseEllipses :: Int -> Indexers -> Indexers
parseEllipses nDims indexers =
  case elemIndices Ell indexers of
    [] ->
      indexers ++ replicate (nDims - length indexers) A
    [dim] ->
      take dim indexers
      ++ replicate (nDims - length indexers + 1) A
      ++ drop (dim + 1) indexers
    _ ->
      error "indexers can only have a single ellipsis \'Ell\'"

-- | Parse negative indexer values.
--
--   Signature: @i -> dim -> indexer -> step -> normIndexer@
normalizeIndexer :: Int -> CLLong -> Indexer -> CLLong -> Indexer
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
parseIndexers :: Shape -> Indexers -> ([Int], Indexers)
parseIndexers shape indexers =
  case V.length shape of {nDims ->
  case parseNone nDims indexers of {(dimsToInsert, indexers) ->
  case parseEllipses nDims indexers of {indexers ->
    if length indexers <= nDims then
      (dimsToInsert,
      zipWith3 (
        \ axis dim indexer ->
          case indexer of
            I _ :. step ->
              normalizeIndexer axis (fromIntegral dim) indexer 1
            indexer :. step ->
              normalizeIndexer axis (fromIntegral dim) indexer step
            _ ->
              normalizeIndexer axis (fromIntegral dim) indexer 1
      ) [0 .. V.length shape - 1]
      (V.toList shape) indexers)
    else
      error
      $ "too many indices for tensor: tensor is "
      ++ show nDims
      ++ "-dimensional, but "
      ++ show (length indexers)
      ++ " were indexed"
  }}}

-- | Validate tensor index correctness.
-- validateTensorIndex :: TensorIndex -> Bool
-- validateTensorIndex = allEqual . Prelude.map shape

-- -- | Integer tensor indexing (advanced indexing).
-- --
-- --   Negative values in index tensors are supported.
-- (!.) :: (HasDtype t) => Tensor t -> TensorIndex -> Tensor t
-- (!.) x tensorIndex
--   | validateTensorIndex tensorIndex =
--     tensor (shape $ head tensorIndex) (
--       \ index ->
--         x ! []
--     )
--   | otherwise =
--     error
--     $ "incorrect tensor index "
--     ++ show (Prelude.map shape tensorIndex)
--     ++ " for shape "
--     ++ show (shape x)

-- | Perform slicing and advanced indexing.
--
--   Negative values in indexer bounds and index tensors are supported.
(!:) :: (HasDtype t) => Tensor t -> Indexers -> Tensor t
(!:) x@(Tensor shape stride offset dat) indexers =
  case parseIndexers shape indexers of {(dimsToInsert, indexers) ->
    insertDims (Tensor (V.fromList $ Prelude.map (
      \ (I start :. end :. step) -> fromIntegral $ (end - start) `Prelude.div` step
    ) $ filter (
      \case
        I _ -> False
        _   -> True
    ) indexers)
    (V.fromList $ Prelude.map (
      \ (_ :. _ :. step, stride) ->
        stride * step
    ) $ filter (
      \ (indexer, stride) ->
        case indexer of
          I _ -> False
          _   -> True
    ) $ zip indexers $ V.toList stride)
    ((+) offset $ Prelude.sum $ zipWith (
      \ indexer stride ->
        case indexer of
          I start :. end :. step ->
            fromIntegral $ start * stride
          I i ->
            fromIntegral $ i * stride
    ) indexers $ V.toList stride) dat) dimsToInsert
  }

{-# INLINE parseNone #-}
{-# INLINE parseEllipses #-}
{-# INLINE normalizeIndexer #-}
{-# INLINE parseIndexers #-}
{-# INLINE (!:) #-}
