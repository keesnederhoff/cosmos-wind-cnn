#!/bin/bash
# Cancel every job the v4 chain submitted (ids recorded by submit_v4_chain.sh).
cd /home/cnederhoff/cosmos/cosmos-wind-cnn
[ -f v4_chain_jobids.txt ] || { echo "no v4_chain_jobids.txt"; exit 1; }
ids=$(awk '{print $2}' v4_chain_jobids.txt | tr '\n' ' ')
echo "cancelling: $ids"
scancel $ids
