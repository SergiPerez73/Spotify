#!/bin/bash

export TOKEN="BQCSK_fy-Zv8nf1KpaPHoVe_U3rYsSYfh_hUWtTRAri2q4f2g-V814Zqxg9Np2F3uZos3i5sqkQ7T--4YHW11xlEg3Gjvw022WF0zvJ0GmGp64ht6sQ7C0YSmZKndT4xUmB8PWakStJSrBFs8mQrj5hrmE8xSF0fx41fuzoeJsY4mxIo49Gb0vWeM2QmUKZlaiV-ztVRQgQBU7kMPXo"
export CLIENT_ID="16c7c999caa041fe96aa9e01c2abf86d"
export CLIENT_SECRET="4e10f9170b4444f2889bab3497806147"
export TEST_SIZE=0.01
export LR=0.1
export N_EPOCHS=100
export PRINT_FREQ=2
export TEST_FREQ=2
export MODEL_PATH=""
export N_BATCHES=10

# python3 CreateDataset.py -client_id="$CLIENT_ID" -client_secret="$CLIENT_SECRET" -token="$TOKEN"

# python3 PreprocessDataset.py

# python3 PreprocessFewSongs.py -client_id="$CLIENT_ID" -client_secret="$CLIENT_SECRET" -token="$TOKEN"

python3 SSRM_pytorch.py -test_size="$TEST_SIZE" -lr="$LR" -n_epochs="$N_EPOCHS" -print_freq="$PRINT_FREQ" -test_freq="$TEST_FREQ" -model_path="$MODEL_PATH" -n_batches="$N_BATCHES"