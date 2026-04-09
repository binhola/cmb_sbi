#!/bin/bash

# ==========================================
# Script to rsync simulations_10000 to Perlmutter
# ==========================================

# Remote username and host
REMOTE_USER="binguyen"
REMOTE_HOST="perlmutter.nersc.gov"

# Remote destination folder
REMOTE_DIR="/pscratch/sd/b/binguyen/cmb_sbi/simulations_10000"

# Local folder to send
LOCAL_DIR="simulations_10000/"

# Make sure the remote folder exists
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR && chmod u+rwx $REMOTE_DIR"

# rsync with progress and compression
rsync -avP -z $LOCAL_DIR $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR

echo "Transfer complete!"
