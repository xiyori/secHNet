module Data.Tensor.Conv where

import Data.Vector.Storable (Storable, Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VM
import Data.Tensor.PlainIndex
import Data.Tensor.Size
import Data.Tensor.Definitions
import Data.Tensor.Functional as T
import Data.Tensor.Instances
import qualified Data.Tensor.Boolean as T
import Data.Tensor.AdvancedIndex

import System.IO.Unsafe
import Foreign.C.Types

-- | Return a sliding window view of tensor.
--
--   Signature: @(shape, stride) -> windowShape -> step -> (shape, stride)@
viewAsWindows :: (Shape, Stride) -> Shape -> Index -> (Shape, Stride)
viewAsWindows (shape, stride) windowShape step =
  case V.length shape of {nDims ->
  case (
    if V.length windowShape == 1 then
      full [nDims] $ fromIntegral $ V.head windowShape
    else if V.length windowShape == nDims then
      fromVector $ V.map fromIntegral windowShape
    else
      error
      $ "viewAsWindows: window shape "
      ++ show windowShape
      ++ " is incompatible with tensor of shape "
      ++ show shape
  ) of {windowShape ->
  case (
    if length step == 1 then
      if head step >= 1 then
        replicate nDims $ head step
      else error "viewAsWindows: step must be >= 1"
    else if length step == nDims then
      step
    else
      error
      $ "viewAsWindows: step "
      ++ show step
      ++ " is incompatible with tensor of shape "
      ++ show shape
  ) of {step ->
  case (fromVector $ V.map fromIntegral shape :: Tensor CLLong) of {tensorShape ->
    if T.all $ tensorShape - windowShape T.>= 0 then
      if T.all $ windowShape - 1 T.>= 0 then
        case Prelude.map (A :. ) step of {slices ->
        case slice (shape, stride, 0) slices of {(_, indexingStride, _) ->
        case ((tensorShape - windowShape) // fromList (Prelude.map fromIntegral step)) + 1
        of {windowIndicesShape ->
          (V.map fromIntegral (toVector windowIndicesShape)
           V.++ V.map fromIntegral (toVector windowShape),
          indexingStride V.++ stride)
        }}}
      else
        error
        $ "viewAsWindows: window shape "
        ++ show (toVector windowShape)
        ++ " too small"
    else
      error
      $ "viewAsWindows: window shape "
      ++ show (toVector windowShape)
      ++ " too large for tensor of shape "
      ++ show shape
  }}}}

-- | Perform n-dimensional batch convolution via stride trick.
--
--   Signature: @tensor -> kernel -> step -> tensor@
conv :: (HasDtype t, Floating t) => Tensor t -> Tensor t -> Index -> Tensor t
conv x@(Tensor shape stride offset dat) kernel@(Tensor kernelShape _ _ _) step =
  case V.length shape of {nDims ->
  case V.length kernelShape of {kernelNDims ->
  case viewAsWindows (V.drop (nDims - kernelNDims) shape, V.drop (nDims - kernelNDims) stride)
  kernelShape step of {(newShape, newStride) ->
  case Tensor (V.take (nDims - kernelNDims) shape V.++ newShape)
  (V.take (nDims - kernelNDims) stride V.++ newStride) offset dat of {memStridedMat ->
  case take nDims $ T.shape memStridedMat of {outputShape ->
    ((memStridedMat `view` [-1, numel kernel]) @ flatten kernel) `view` outputShape
  }}}}}

{-# INLINE viewAsWindows #-}
{-# INLINE conv #-}
