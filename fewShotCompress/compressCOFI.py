# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.
# This file is coded based on train_matchingNet.py.
# coding=utf-8
import torch
import argparse
import time
from utils.transformers import AutoTokenizer, AutoConfig

from utils.models import IntentBERTCoFi, L0Module, LinearClsfier
from utils.IntentDataset import IntentDataset
from utils.Evaluator import FewShotEvaluator, InTaskCoFiModelEvaluator
from utils.Trainer import CoFiFewShotTrainer
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.DataAugmenter import DataAugmenter
from utils.Logger import logger
from torch.utils.data import DataLoader, TensorDataset
import copy
import pdb
import os

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--basemodel', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--augNum', default=1, type=int)
    parser.add_argument('--dataAugCachePath', default=SAVE_PATH, help="Augmented data cache path.")

    # ==== dataset ====
    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')
    
    # ==== evaluation ====
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--learningRate', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)
    parser.add_argument('--testRepetition', default=5, type=int)
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--modelDir', default=SAVE_PATH, help="Model loading directory.")
    parser.add_argument('--wandb', help='use wandb or not', action="store_true")
    parser.add_argument('--wandbProj', help='wandb project name')

    # ====== CoFi  =======
    parser.add_argument('--prepruning_finetune_epochs', type=int, default=1)
    parser.add_argument('--lagrangian_warmup_epochs', type=int, default=2)
    parser.add_argument('--pruning_epochs', type=int, default=20)
    parser.add_argument('--droprate_init', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2/3)
    parser.add_argument('--target_sparsity', type=float, default=0.95)
    parser.add_argument('--pruning_type', default='structured_heads+structured_mlp+hidden+layer')
    parser.add_argument('--saveModel', action='store_true')
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model" 
                        "If none, then there will be a specific name controlled by how the model is trained")

    args = parser.parse_args()

    return args

def main():
    # ======= global env =====
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ======= process arguments ======
    args = parseArgument()
    print(args)

    # ==== setup logger ====
    if args.loggingLevel == LOGGING_LEVEL_INFO:
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)

    # ==== set seed ====
    if args.seed >= 0:
        set_seed(args.seed)
        logger.info("The random seed is set %d"%(args.seed))
    else:
        logger.info("The random seed is not set any value.")

    # ======= process data ======
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # load raw dataset
    logger.info("----- Testing Data -----")
    logger.info(f"Loading data from {args.testDataset}")
    dataset = IntentDataset(args.testDataset, mode=DATASET_MODE_PARTITION_MERGE_INTO_TEST)
    dataset = dataset.selectByDomains(splitName(args.testDomain))
    dataset.tokenize(tok)

    # ======= prepare model ======
    # initialize model
    logger.info("----- IntentBERTCoFi initializing -----")
    modelConfig = {}
    device = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['device'] = device
    modelConfig['clsNumber'] = 90
    modelConfig['LMName'] = args.LMName
    modelConfig['basemodel'] = args.basemodel
    modelConfig['l0_module_droprate_init'] = args.droprate_init
    modelConfig['l0_module_temperature'] = args.temperature
    modelConfig['l0_module_target_sparsity'] = args.target_sparsity
    modelConfig['l0_module_pruning_type'] = args.pruning_type
    modelConfig['modelDir'] = args.modelDir
    # model = IntentBERTCoFi(modelConfig)

    # setup evaluator
    testParam = {'batch_size': args.batch_size}
    tester = InTaskCoFiModelEvaluator(testParam, dataset)
    validator = copy.deepcopy(tester)

    # setup trainer for pruning
    trainingParam = {"shot":args.shot,  \
            'batch_size': args.batch_size,  \
            'lr': args.learningRate,  \
            'weight_decay': args.weightDecay, \
            'prepruning_finetune_epochs':args.prepruning_finetune_epochs,  \
            'lagrangian_warmup_epochs':args.lagrangian_warmup_epochs,  \
            'pruning_epochs': args.pruning_epochs, \
            "wandb": args.wandb, \
            "wandbProj": args.wandbProj, \
            "wandbRunName": "None", \
            "wandbConfig": {}
            }
    coFiTrainer = CoFiFewShotTrainer(trainingParam, dataset, validator)

    # pruning and testing
    performList = []   # acc, pre, rec, fsc
    for repetitionID in range(args.testRepetition):
        logger.info(f'Repeating {repetitionID}/{args.testRepetition}.')

        # update trainer
        coFiTrainer.setWandbRunName("sd%d-data%s-sp%.2f-st%d-PEpoch%d-rep%d"%(args.seed, args.testDataset, args.target_sparsity, args.shot, args.pruning_epochs, repetitionID))

        # initialize the model
        model = IntentBERTCoFi(modelConfig)

        # initialize linear classifier
        lcParamPath = os.path.join(args.modelDir, args.LMName, SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE)
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None, 'loadModelPath':lcParamPath}
        model.linearClsfier = LinearClsfier(lcConfig)
        model.linearClsfier.to(model.device)

        # sample K-shot data
        logger.info("Sampling K-shot data ...")
        set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.
        tensorDataset, uttListFewShot, labList = dataset.trainPart.randomSliceTorchDataset(args.shot, tok, returnUttListLabList=True)
        kShotDataLoader = DataLoader(tensorDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        logger.info(f"Sliced {args.shot} data from dataset. K-shot data size is {len(tensorDataset)}. Signature: {getMD5(uttListFewShot)}.")

        # sample some unlabeled data for model compression
        mode = 2
        if mode == 1:   # sample from original dataset
            tensorDatasetUnlabel = dataset.trainPart.randomSliceUnlabeledTorchDataset(args.unlabDataCountPerLab, tok, exceptUttList=uttListFewShot, mergeExcept = True)
            if not tensorDatasetUnlabel == None:
                dataloaderUnlabel = DataLoader(tensorDatasetUnlabel, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
                logger.info(f"Unlabeled data: {len(tensorDatasetUnlabel)} data.")
            else:
                dataloaderUnlabel = None
        elif mode == 2:   # use generated data
            dataAugmenterParam={"cachePath": f"{args.dataAugCachePath}", \
                    "cacheFile": f"DS{args.testDataset}_SE{args.seed}_ST{args.shot}_DN{args.augNum}.pk"}
            dataAugmenter = DataAugmenter(dataAugmenterParam)
            unlabTensorData, newUttList, newLabList = dataAugmenter.dataAugContextGPT3(uttListFewShot, labList, augNum = args.augNum, device=model.device, returnTensorDataset = True, mergeOriData = True, oriTensorDataset = tensorDataset, dataset=dataset, tokenizerOri = tok)
            dataloaderUnlabel = DataLoader(unlabTensorData, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        else:
            logger.info(f"Invalid mode {mode}.")

        # evaluate before pruning: sampled K-shot data are used to build up a linear classifier
        logger.info("Evaluating model before fine-tuning and pruning ...")
        acc, pre, rec, F1 = tester.evaluate(model, tok, kShotDataLoader)
        logger.info(f'Before pruning, ')
        logger.info(f'acc - {acc}')
        logger.info(f'pre - {pre}')
        logger.info(f'rec - {rec}')
        logger.info(f'F1  - {F1}')

        # prune model
        # coFiTrainer.train(model, tok, kShotDataLoader)
        modelWithTrainedMasks, modelPruned = coFiTrainer.train(model, tok, dataloaderUnlabel)

        # evaluate
        # logger.info("Evaluating model ...")
        acc, pre, rec, F1 = tester.evaluate(model, tok, kShotDataLoader)
        performList.append([acc, pre, rec, F1])

    # performance mean and std
    performMean = np.mean(np.stack(performList, 0), 0)
    performStd  = np.std(np.stack(performList, 0), 0)
    itemList = ["acc", "pre", "rec", "fsc"]
    logger.info("Evaluate statistics: ")
    printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

    # save model in disk
    if args.saveModel:
        # create folder if not exists
        save_path = os.path.join(SAVE_PATH, args.saveName)
        if not os.path.exists(save_path):
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            logger.info(f"The new directory is created: {save_path}.")

        # decide the save name
        save_path = os.path.join(SAVE_PATH, args.saveName, SAVE_STATE_DICT_FILE_NAME)
        # save
        torch.save(modelWithTrainedMasks.state_dict(), save_path)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
