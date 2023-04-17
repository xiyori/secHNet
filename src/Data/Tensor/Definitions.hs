module Data.Tensor.Definitions where

import Data.Vector.Storable (Storable, Vector)
import Data.Tensor.Index

import Foreign.C.Types

-- | Typeclass for data types which can be stored in tensors.
class (Storable t) => HasDtype t where
  tensorDtype :: Tensor t -> CInt

  showDtype :: Tensor t -> String

-- | Tensor data type.
data (HasDtype t) =>
  Tensor t = Tensor {
    -- | Tensor shape.
    shape :: !Index,
    -- | Data stride in bytes, analogous to NumPy array stride.
    tensorStride :: !Stride,
    -- | Data offset in bytes.
    tensorOffset :: !CSize,
    -- | Internal data representation.
    tensorData :: !(Vector t)
  }

-- | Advanced indexer data type.
type TensorIndex = [Tensor CLLong]
