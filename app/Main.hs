module Main where

import Bank.Error (BankError)
import Bank.Handle (Account, Balance, BankHandle (BankHandle), HasBank, newBankHandle)
import Data.IORef (readIORef)
import Control.Monad (forever)
import Control.Monad.Trans.Reader (ReaderT)

data Input
  = INewAccount Account
  | IDeleteAccount Account
  | ITransfer Account Balance Account
  deriving (Read)

-- (0.5 балла) Реализуйте вспомогательные функции для взаимодействия с
-- пользователем.

parse :: IO Input
-- ^ Reads next command from standard input
parse = getLine >>= readIO

printStorage :: BankHandle -> IO ()
-- ^ Outputs bank storage contents
printStorage (BankHandle h) = readIORef h >>= print

-- (1 балл) Реализуйте взаимодействие с пользователем, использующее
-- вспомогательные функции выше и интерфейс BankHandle.
--
-- Рекомендуется использовать функцию @forever@ из модуля @Control.Monad@.

withRunInIO f = do
  h <- asks getBankHandle
  lift (f runReaderT h)

main :: IO ()
main = new >>= runReaderT app
  where
    app = forever $ withRunInIO $ \runInIO ->
      handle (print :: BankError -> IO ()) $
      handle (print :: IOError -> IO ()) $
      runInIO $ do 
        cmd <- lift parse
        case cmd of
          INewAccount -> do
            acc <- newAccount
            deposit acc 100
          IDeleteAccount acc -> deleteAccount acc
          ITransfer from amount to -> transfer from (Sum amount) to
        h <- asks getBankHandle
        lift $ printStorage h
    
