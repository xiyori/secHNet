module Data.Tensor.Size where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V

import Foreign
import Foreign.C.Types

sizeOfElem :: Storable t => Vector t -> CSize
sizeOfElem = fromIntegral . sizeOf . V.head
