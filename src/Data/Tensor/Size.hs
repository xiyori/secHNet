module Data.Tensor.Size where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V

import Foreign
import Foreign.C.Types

sizeOfCBool :: CInt
sizeOfCBool = fromIntegral $ sizeOf (0 :: CBool)

sizeOfCChar :: CInt
sizeOfCChar = fromIntegral $ sizeOf (0 :: CChar)

sizeOfCInt :: CInt
sizeOfCInt = fromIntegral $ sizeOf (0 :: CInt)

sizeOfCLong :: CInt
sizeOfCLong = fromIntegral $ sizeOf (0 :: CLong)

sizeOfCLLong :: CInt
sizeOfCLLong = fromIntegral $ sizeOf (0 :: CLLong)

sizeOfCFloat :: CInt
sizeOfCFloat = fromIntegral $ sizeOf (0 :: CFloat)

sizeOfCDouble :: CInt
sizeOfCDouble = fromIntegral $ sizeOf (0 :: CDouble)

sizeOfElem :: Storable t => Vector t -> CInt
sizeOfElem = fromIntegral . sizeOf . V.head
