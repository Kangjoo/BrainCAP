REPO_DIR=/Users/sab322/dev/brainCAP
ENV_DIR=/Users/sab322/anaconda3/envs

## FIRST TIME ONLY ##

#Create conda env
conda env create -f ${REPO_DIR}/environment_linux.yml

#Make wrapper executable
chmod 770 ${REPO_DIR}/brainCAP/braincap.py

#Optionally, add to PATH. You can add this to your shell profile
export PATH="${REPO_DIR}/brainCAP:$PATH"

## END FIRST TIME ONLY ##

#Source the brainCAP environment
source activate ${ENV_DIR}/brainCAP

${REPO_DIR}/brainCAP/braincap.py \
    --config="${REPO_DIR}/tutorials/OHBM2025/braincap_demo.yaml" \
    --steps="concatenate_bolds" \
    --dryrun="yes" 

${REPO_DIR}/brainCAP/braincap.py \
    --config="${REPO_DIR}/tutorials/OHBM2025/braincap_demo.yaml" \
    --steps="prep" \
    --dryrun="yes" 

${REPO_DIR}/brainCAP/braincap.py \
    --config="${REPO_DIR}/tutorials/OHBM2025/braincap_demo.yaml" \
    --steps="clustering" \
    --dryrun="yes" 

${REPO_DIR}/brainCAP/braincap.py \
    --config="${REPO_DIR}/tutorials/OHBM2025/braincap_demo.yaml" \
    --steps="post" \
    --dryrun="yes" 


${REPO_DIR}/brainCAP/braincap.py \
    --config="${REPO_DIR}/tutorials/OHBM2025/braincap_demo.yaml" \
    --steps="concatenate_bolds,prep,clustering,post" \
    --dryrun="yes" 