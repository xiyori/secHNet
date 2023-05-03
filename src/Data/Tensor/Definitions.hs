{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleInstances #-}

module Data.Tensor.Definitions where

import Data.Vector.Storable (Storable, Vector)

import Foreign.C.Types

-- | Typeclass for data types which can be stored in tensors.
class (Storable t) => HasDtype t where
  tensorDtype :: Tensor t -> CInt

  showDtype :: Tensor t -> String

class (HasDtype t, Num t) => HasArange t where
  -- | Return evenly spaced values within a given interval.
  --
  --   Example: @arange 0 3 1 = tensor([0, 1, 2])@
  --
  --   Signature: @low -> high -> step -> tensor@
  arange :: t -> t -> t -> Tensor t

-- | Tensor data type.
data (HasDtype t) =>
  Tensor t = Tensor {
    -- | Tensor shape.
    shape :: !Shape,
    -- | Data stride in bytes, analogous to NumPy array stride.
    tensorStride :: !Index,
    -- | Data offset in bytes.
    tensorOffset :: !CSize,
    -- | Internal data representation.
    tensorData :: !(Vector t)
  }

-- | Tensor shape @Vector CSize@.
type Shape = Vector CSize

-- | Tensor index and stride @Vector CLLong@.
type Index = Vector CLLong

-- | Indexer data type.
data Indexer
  -- | Single index @I index@.
  = I CLLong
  -- | Full slice, analogous to NumPy @:@.
  | A
  -- | Slice from start, analogous to NumPy @start:@.
  | S CLLong
  -- | Slice till end, analogous to NumPy @:end@.
  | E CLLong
  | Indexer -- | Slice @I start :. end@, @S start :. step@,
          --   @E end :. step@ or @start :. end :. step@,
          --   analogous to NumPy @start:end:step@.
          :. CLLong
  -- | Tensor index (advanced indexing).
  | T (Tensor CLLong)
  -- | Insert new dim, analogous to NumPy @None@.
  | None
  -- | Ellipses, analogous to NumPy @...@.
  | Ell

infixl 5 :.

-- | Slice indexer data type.
type Indexers = [Indexer]
