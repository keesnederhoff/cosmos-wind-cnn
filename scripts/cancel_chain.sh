#!/bin/bash
# Cancel every job the chain submitted (ids recorded by submit_chain.sh).
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
[ -f chain_jobids.txt ] || { echo "no chain_jobids.txt"; exit 1; }
ids=$(awk '{print $2}' chain_jobids.txt | tr '\n' ' ')
echo "cancelling: $ids"
scancel $ids
