export TOKEN=""
export CLIENT_ID=""
export CLIENT_SECRET=""
export TEST_SIZE=0.01
export LR=0.1
export N_EPOCHS=1000
export PRINT_FREQ=2
export TEST_FREQ=2
export MODEL_PATH=''
export N_BATCHES=10

# python3 CreateDataset.py -client_id="$CLIENT_ID" -client_secret="$CLIENT_SECRET" -token="$TOKEN"

# python3 PreprocessDataset.py

# python3 PreprocessFewSongs.py -client_id="$CLIENT_ID" -client_secret="$CLIENT_SECRET" -token="$TOKEN"

python3 SSRM_pytorch.py -test_size="$TEST_SIZE" -lr="$LR" -n_epochs="$N_EPOCHS" -print_freq="$PRINT_FREQ" -test_freq="$TEST_FREQ" -model_path="$MODEL_PATH" -n_batches="$N_BATCHES"