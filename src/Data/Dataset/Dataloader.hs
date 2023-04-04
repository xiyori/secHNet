module Data.Dataset.Dataloader where

import Data.Dataset.Dataset
import Data.Conduit

randomSample :: (Dataset s m t) => s -> ConduitT () t m ()
randomSample = error "TODO"

-- TODO
-- Обработка данных это тоже кондуиты

toBatch :: ConduitT t [t] m ()
toBatch = error "TODO"