# Environment Setup
    conda env create -f conda-env_20220728_transformer4.20.yml
    pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Script to train teacher model
    - to generate few shot data: ./teacherTrain/scripts/fewShotFineTune_genFewData.sh
    - to train the teacher model: ./teacherTrain/scripts/fewShotFineTune.sh

# Script to compress the neural part of the model
    ./fewShotCompress/scripts/compress_CoFi.sh
    
# Script to prune the vocabulary
    - To select the top-k tokens: ./vocabPrune/scripts/extractStudentVocab.sh
    - To conduct PCA: ./evaluation/scripts/vocabPCACompress.sh

# Script for evaluation
    ./evaluation/scripts/evalCompressed.sh
