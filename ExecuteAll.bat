@echo off
set TOKEN=token
set CLIENT_ID=client_id
set CLIENT_SECRET=client_secret
set TEST_SIZE=0.15
set LR=0.1
set N_EPOCHS=2000
set PRINT_FREQ=50
set TEST_FREQ=5
set MODEL_PATH=""
set N_BATCHES=10

REM python.exe CreateDataset.py -client_id=%CLIENT_ID% -client_secret=%CLIENT_SECRET% -token=%TOKEN%

REM python.exe PreprocessDataset.py

REM python.exe PreprocessFewSongs.py -client_id=%CLIENT_ID% -client_secret=%CLIENT_SECRET% -token=%TOKEN%

python.exe SSRM_pytorch.py -test_size=%TEST_SIZE% -lr=%LR% -n_epochs=%N_EPOCHS% -print_freq=%PRINT_FREQ% -test_freq=%TEST_FREQ% -model_path=%MODEL_PATH% -n_batches=%N_BATCHES%