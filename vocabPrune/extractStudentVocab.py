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
import pickle
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
    parser.add_argument('--vocabKeepSize', type=int, default=0)

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

    # pruning and testing
    performList = []   # acc, pre, rec, fsc
    for repetitionID in range(args.testRepetition):
        logger.info(f'Repeating {repetitionID}/{args.testRepetition}.')

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
            vocab = dataAugmenter.extractSubVocab(uttListFewShot, labList, augNum = args.augNum, device='cpu', returnTensorDataset = True, mergeOriData = True, oriTensorDataset = tensorDataset, dataset=dataset, tokenizerOri = tok, vocabKeepSize = args.vocabKeepSize)
        else:
            logger.info(f"Invalid mode {mode}.")

    # save vocab
    vocabDir = './vocab'
    if not os.path.exists(vocabDir):
        os.makedirs(vocabDir)
    cacheFilePath = os.path.join(vocabDir, f"vocab{args.testDataset}_SE{args.seed}_ST{args.shot}_DN{args.augNum}_KS{args.vocabKeepSize}.pk")
    logger.info(f"Saving cache: {cacheFilePath}")
    result = {"RESULT": list(vocab)}
    with (open(cacheFilePath, "wb")) as openfile:
        pickle.dump(result, openfile)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
