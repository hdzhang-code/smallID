echo usage:
echo scriptName.sh : run in normal mode
echo scriptName.sh debug : run in debug mode

# hardware
cudaID=$2

# debug mode
if [[ $# != 0 ]] && [[ $1 == "debug" ]]
then
    debug=true
else
    debug=false
fi

seed=1

# wandb
# wandb=--wandb
wandb=
wandbProj=temp

# dataset
testDataName=(bank77 mcid HINT3 OOS hwu64_publishedPaper)

# evaluation setting
shot=2
shotList=(2 5 10)
shotList=(5)
learningRate=2e-5
weightDecay=0.001
saveModel=--saveModel

# pruning
prepruning_finetune_epochs=100
pruning_epochs=2000
pruning_epochs=1000
testRepetition=1
target_sparsityList=(0.95 0.85)
target_sparsityList=(0.95)
target_sparsityList=(0.97 0.99)
target_sparsityList=(0.95)
target_sparsity=0.50
lagrangian_warmup_epochs=10

vocabKeepSizeList=(500 1000 1500 2000 4000 8000 16000 30522)
vocabKeepSizeList=(2000)
vocabKeepSize=1

augNum=50
dataAugCachePath=../dataAugCache

# model initialization
basemodel=bert-base-uncased
tokenizer=bert-base-uncased

seedList=(1 2 3 4 5)
seedList=(5)

modelDir='../teacherTrain/saved_models/'

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    # epochs=1
    wandb=
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
for seed in ${seedList[@]}
do
    for testData in ${testDataName[@]}
    do
        for shot in ${shotList[@]}
        do
            for target_sparsity in ${target_sparsityList[@]}
            do
                for vocabKeepSize in ${vocabKeepSizeList[@]}
                do
                    case ${testData} in
                        bank77)
                            testDataset=bank77
                            testDomain="BANKING"
                            ;;
                        mcid)
                            testDataset=mcid
                            testDomain="MEDICAL"
                            ;;
                        HINT3)
                            testDataset=HINT3
                            testDomain='curekart,powerplay11,sofmattress'
                            ;;
                        OOS)
                            testDataset=OOS
                            testDomain='travel,kitchen_dining'
                            ;;
                        hwu64_publishedPaper)
                            testDataset=hwu64_publishedPaper
                            testDomain='play,lists,recommendation,iot,general,transport,weather,social,email,music,qa,takeaway,audio,news,datetime,calendar,cooking,alarm'
                            ;;
                        *)
                            echo Invalid testData ${testData}
                    esac

                    LMName=FewShotFineTuneModel_seed${seed}_testD${testData}_${shot}shot_LMbert-base-uncased_lrBb2e-4_lrCls2e-5
                    # LMName=bert-base-uncased
                    logFolder=./log/
                    mkdir -p ${logFolder}
                    logFile=${logFolder}/compCoFi_testD${testData}_${shot}shot_$LM${LMName}_sp${target_sparsity}_seed${seed}_sp${target_sparsity}.log 
                    if $debug
                    then
                        logFlie=${logFolder}/logDebug.log
                    fi
                    saveName=compre_${LMName}_sp${target_sparsity}

                    export CUDA_VISIBLE_DEVICES=${cudaID}
                    python extractStudentVocab.py  \
                        --seed ${seed} \
                        --testDataset ${testDataset} \
                        --testDomain ${testDomain} \
                        --tokenizer  ${tokenizer}   \
                        --shot ${shot}  \
                        --LMName ${LMName} \
                        --learningRate  ${learningRate}  \
                        --weightDecay  ${weightDecay} \
                        --prepruning_finetune_epochs  ${prepruning_finetune_epochs}  \
                        --basemodel  ${basemodel}  \
                        --pruning_epochs  ${pruning_epochs}  \
                        --testRepetition  ${testRepetition}  \
                        --target_sparsity   ${target_sparsity} \
                        ${wandb}  \
                        --wandbProj  ${wandbProj}  \
                        --lagrangian_warmup_epochs   ${lagrangian_warmup_epochs}   \
                        --augNum   ${augNum} \
                        --dataAugCachePath   ${dataAugCachePath}  \
                        --modelDir   ${modelDir}  \
                        ${saveModel} \
                        --saveName ${saveName} \
                        --vocabKeepSize   ${vocabKeepSize} \
                        --disableCuda  \
                        # | tee "${logFile}"
                    done
                done
            done
        done
    done
echo "Experiment finished."
