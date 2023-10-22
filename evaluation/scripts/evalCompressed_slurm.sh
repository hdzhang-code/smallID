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

# dataset
testDataName=(bank77 mcid HINT3 OOS hwu64_publishedPaper)

# evaluation setting
shot=2
shotList=(1 2 5 10)
shotList=(2 5 10)
shotList=(5)
testRepetition=1

target_sparsityList=(0.99)
target_sparsityList=(0.95)
target_sparsityList=(0.97 0.99)
target_sparsityList=(0.95)

alphaList=(0.1 0.2 0.4 0.8 1.6 3.2 6.4)
alphaList=(3.2 6.4)
alphaList=(0.1 0.4 0.8)
alphaList=(0.6 1.0 1.2)
alphaList=(1.4)
alphaList=(0.6 1.0 1.2)
alphaList=(1.2)

# model initialization
tokenizer=bert-base-uncased

lrBackboneListName=(2e-5 2e-4 2e-3)
lrBackboneListName=(2e-4)
lrClsfierListName=(2e-5 2e-4 2e-3 2e-2)
lrClsfierListName=(1e-5 5e-6)
lrClsfierListName=(2e-5)
sdWeightListName=(0.0)

modelDir=../fewShotCompress/saved_models/

modelLayer=1

distill_temperatureList=(0.001 0.01 0.1 1.0 10 100 1000)
distill_temperatureList=(10)

vocabKeepSizeList=(500 1000 1500 2000 4000 8000 16000 30522)
vocabKeepSizeList=(500 1000 1500 2000 4000 8000 16000)
vocabKeepSizeList=(4000)
seedModelName=(1 2 3 4 5)

pcaDimList=(100 200 300 400 500 600 700 760)
pcaDimList=(400)

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=1
    wandb=
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
for distill_temperature in ${distill_temperatureList[@]}
do
    for testData in ${testDataName[@]}
    do
        for shot in ${shotList[@]}
        do
            for target_sparsity in ${target_sparsityList[@]}
            do
                for seedName in ${seedModelName[@]}
                do
                    for lrBackboneName in ${lrBackboneListName[@]}
                    do
                        for lrClsfierName in ${lrClsfierListName[@]}
                        do
                            for sdWeightName  in ${sdWeightListName[@]}
                            do
                                for vocabKeepSize in ${vocabKeepSizeList[@]}
                                do
                                    for alpha in ${alphaList[@]}
                                    do
                                    for pcaDim in ${pcaDimList[@]}
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

                            # LMName=compre_FewShotFineTuneModel_seed${seedName}_testD${testData}_${shot}shot_LMbert-base-uncased_lrBb${lrBackboneName}_lrCls${lrClsfierName}_ML${modelLayer}_UL0_AN50_T${distill_temperature}
                            LMName=compre_FewShotFineTuneModel_seed${seedName}_testD${testData}_${shot}shot_LMbert-base-uncased_lrBb${lrBackboneName}_lrCls${lrClsfierName}_sp${target_sparsity}
                            logFolder=./log/
                            mkdir -p ${logFolder}
                            logFile=${logFolder}/evalFewShotFT_${LMName}_KS${vocabKeepSize}_pca${pcaDim}.log
                            if $debug
                            then
                                logFlie=${logFolder}/logDebug.log
                            fi

                            externalVocabFile=vocab${testData}_SE${seedName}_ST${shot}_DN50_KS${vocabKeepSize}.pk

                            pcaVocabFileName=D${testDataset}_SD${seedName}_ST${shot}_pca${pcaDim}.pk

                            export CUDA_VISIBLE_DEVICES=${cudaID}
                            python evalCompressed.py \
                                --seed ${seed} \
                                --testDataset ${testDataset} \
                                --testDomain ${testDomain} \
                                --tokenizer  ${tokenizer}   \
                                --LMName ${LMName} \
                                --target_sparsity   ${target_sparsity} \
                                ${wandb}  \
                                --modelLayer   ${modelLayer}  \
                                --externalVocabFile   ${externalVocabFile}  \
                                --modelDir  ${modelDir} \
                                --pcaVocabFileName  ${pcaVocabFileName}  \
                                | tee "${logFile}"
                            done
                        done
                    done
                done
            done
        done
    done
done
done
done
done
echo "Experiment finished."
