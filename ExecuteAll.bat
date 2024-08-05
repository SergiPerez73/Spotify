@echo off
set TOKEN=
set CLIENT_ID=16c7c999caa041fe96aa9e01c2abf86d
set CLIENT_SECRET=4e10f9170b4444f2889bab3497806147
set TEST_SIZE=0.1
set LR=0.1
set N_EPOCHS=20000
set PRINT_FREQ=1000
set TEST_FREQ=2000
set MODEL_PATH=""
set N_BATCHES=10

python.exe CreateDataset.py -client_id=%CLIENT_ID% -client_secret=%CLIENT_SECRET% -token=%TOKEN%

python.exe PreprocessDataset.py

python.exe PreprocessFewSongs.py -client_id=%CLIENT_ID% -client_secret=%CLIENT_SECRET% -token=%TOKEN%

REM python.exe SSRM_pytorch.py -test_size=%TEST_SIZE% -lr=%LR% -n_epochs=%N_EPOCHS% -print_freq=%PRINT_FREQ% -test_freq=%TEST_FREQ% -model_path=%MODEL_PATH% -n_batches=%N_BATCHES%