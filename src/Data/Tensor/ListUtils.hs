module Data.Tensor.ListUtils where

import Data.Vector.Storable (Storable, Vector, (!), (//))
import qualified Data.Vector.Storable as V
import Data.Tensor.Index
import Data.Tensor.Definitions

-- | Determine tensor shape from list.
parseShape1 :: [t] -> Index
parseShape1 listData = V.singleton $ fromIntegral $ length listData

-- | Determine tensor shape from list of lists.
parseShape2 :: [[t]] -> Index
parseShape2 listData =
  case map length listData of {lengths0 ->
    if allEqual lengths0 then
      V.fromList [fromIntegral $ length lengths0,
                  fromIntegral $ head lengths0]
    else
      error "list dimensions are not uniform"
  }

-- | Determine tensor shape from list of lists.
parseShape3 :: [[[t]]] -> Index
parseShape3 listData =
  case map length listData of {lengths0 ->
  case concatMap (
    map length
  ) listData of {lengths1 ->
    if allEqual lengths0 &&
       allEqual lengths1 then
      V.fromList [fromIntegral $ length lengths0,
                  fromIntegral $ head lengths0,
                  fromIntegral $ head lengths1]
    else
      error "list dimensions are not uniform"
  }}

-- | Determine tensor shape from list of lists.
parseShape4 :: [[[[t]]]] -> Index
parseShape4 listData =
  case map length listData of {lengths0 ->
  case concatMap (
    map length
  ) listData of {lengths1 ->
  case concatMap (
    concatMap
    $ map length
  ) listData of {lengths2 ->
    if allEqual lengths0 &&
       allEqual lengths1 &&
       allEqual lengths2 then
      V.fromList [fromIntegral $ length lengths0,
                  fromIntegral $ head lengths0,
                  fromIntegral $ head lengths1,
                  fromIntegral $ head lengths2]
    else
      error "list dimensions are not uniform"
  }}}

-- | Determine tensor shape from list of lists.
parseShape5 :: [[[[[t]]]]] -> Index
parseShape5 listData =
  case map length listData of {lengths0 ->
  case concatMap (
    map length
  ) listData of {lengths1 ->
  case concatMap (
    concatMap
    $ map length
  ) listData of {lengths2 ->
  case concatMap (
    concatMap
    $ concatMap
    $ map length
  ) listData of {lengths3 ->
    if allEqual lengths0 &&
       allEqual lengths1 &&
       allEqual lengths2 &&
       allEqual lengths3 then
      V.fromList [fromIntegral $ length lengths0,
                  fromIntegral $ head lengths0,
                  fromIntegral $ head lengths1,
                  fromIntegral $ head lengths2,
                  fromIntegral $ head lengths3]
    else
      error "list dimensions are not uniform"
  }}}}

-- | Determine tensor shape from list of lists.
parseShape6 :: [[[[[[t]]]]]] -> Index
parseShape6 listData =
  case map length listData of {lengths0 ->
  case concatMap (
    map length
  ) listData of {lengths1 ->
  case concatMap (
    concatMap
    $ map length
  ) listData of {lengths2 ->
  case concatMap (
    concatMap
    $ concatMap
    $ map length
  ) listData of {lengths3 ->
  case concatMap (
    concatMap
    $ concatMap
    $ concatMap
    $ map length
  ) listData of {lengths4 ->
    if allEqual lengths0 &&
       allEqual lengths1 &&
       allEqual lengths2 &&
       allEqual lengths3 &&
       allEqual lengths4 then
      V.fromList [fromIntegral $ length lengths0,
                  fromIntegral $ head lengths0,
                  fromIntegral $ head lengths1,
                  fromIntegral $ head lengths2,
                  fromIntegral $ head lengths3,
                  fromIntegral $ head lengths4]
    else
      error "list dimensions are not uniform"
  }}}}}


-- | Convert list to vector.
parseData1 :: Storable t => [t] -> Vector t
parseData1 = V.fromList

-- | Flatten list of lists to vector.
parseData2 :: Storable t => [[t]] -> Vector t
parseData2 listData =
  V.fromList
  $ concat listData

-- | Flatten list of lists to vector.
parseData3 :: Storable t => [[[t]]] -> Vector t
parseData3 listData =
  V.fromList
  $ concat
  $ concat listData

-- | Flatten list of lists to vector.
parseData4 :: Storable t => [[[[t]]]] -> Vector t
parseData4 listData =
  V.fromList
  $ concat
  $ concat
  $ concat listData

-- | Flatten list of lists to vector.
parseData5 :: Storable t => [[[[[t]]]]] -> Vector t
parseData5 listData =
  V.fromList
  $ concat
  $ concat
  $ concat
  $ concat listData

-- | Flatten list of lists to vector.
parseData6 :: Storable t => [[[[[[t]]]]]] -> Vector t
parseData6 listData =
  V.fromList
  $ concat
  $ concat
  $ concat
  $ concat
  $ concat listData

{-# INLINE parseShape1 #-}
{-# INLINE parseShape2 #-}
{-# INLINE parseShape3 #-}
{-# INLINE parseShape4 #-}
{-# INLINE parseShape5 #-}
{-# INLINE parseShape6 #-}
{-# INLINE parseData1 #-}
{-# INLINE parseData2 #-}
{-# INLINE parseData3 #-}
{-# INLINE parseData4 #-}
{-# INLINE parseData5 #-}
{-# INLINE parseData6 #-}
