class HasParams t a where
    getParams :: a -> [Matrix t]
    setParams :: [Matrix t] -> a


