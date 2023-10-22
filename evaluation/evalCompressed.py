# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.

# This file is coded based on train_matchingNet.py.
# coding=utf-8
import torch
import argparse
import time
from utils.transformers import AutoTokenizer

from utils.models import IntentBERT, LinearClsfier, IntentBERTCoFi
from utils.IntentDataset import IntentDataset
from utils.Evaluator import FewShotEvaluator, TestPartitionEvaluator, InTaskCoFiModelEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
import pickle
import pdb
import os

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--basemodel', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--modelLayer', default=12, type=int)

    # ==== dataset ====
    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--pcaVocabFileName', help="None")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')

    # COFI
    parser.add_argument('--droprate_init', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=2/3)
    parser.add_argument('--target_sparsity', type=float, default=0.95)
    parser.add_argument('--pruning_type', default='structured_heads+structured_mlp+hidden+layer')
    
    # ==== evaluation ====
    parser.add_argument('--disableCuda', action="store_true")
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--modelDir', default=SAVE_PATH, help="Model loading directory.")
    parser.add_argument('--externalVocabFile', help="External vocabulary file.")

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

    # read vocab file
    vocabFilePath = os.path.join('../vocabPrune/vocab/', args.externalVocabFile) 
    if os.path.exists(vocabFilePath):
        logger.info(f"Reading vocab from: {vocabFilePath}")
        with (open(vocabFilePath, "rb")) as openfile:
            vocabFile = pickle.load(openfile)
    vocabList = vocabFile['RESULT']
    # vocabList = None
    # dataset.tokenize(tok, vocabList)

    # ======= prepare model ======
    # initialize model: backbone
    logger.info("----- IntentBERTCoFi initializing -----")
    modelConfig = {}
    device = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['device'] = device
    modelConfig['clsNumber'] = 90
    modelConfig['LMName'] = 'bert-base-uncased'
    modelConfig['basemodel'] = 'bert-base-uncased'
    modelConfig['l0_module_droprate_init'] = args.droprate_init
    modelConfig['l0_module_temperature'] = args.temperature
    modelConfig['l0_module_target_sparsity'] = args.target_sparsity
    modelConfig['l0_module_pruning_type'] = args.pruning_type
    model = IntentBERTCoFi(modelConfig)
    model.to(device)
    # linear classifier
    lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None}
    model.linearClsfier = LinearClsfier(lcConfig)
    model.linearClsfier.to(device)

    # load model
    paramPath = os.path.join(args.modelDir, args.LMName, 'save_state_dict_file.pth')
    model.load_state_dict(torch.load(paramPath))
    model.eval()

    # tokenize: replace those tokens not in the vocab with the nearest one in the vocab.
    vocabMap = dataset.buildVocabMap(tok, vocabList, model)
    dataset.tokenize(tok, vocabList, vocabMap)

    # pruning model
    logger.info(f"Pruning model ...")
    modelParamesOri = model.calculate_parameters()
    model.prune_model_with_z()
    modelParames = model.calculate_parameters()
    remainingRatio = modelParames / modelParamesOri
    logger.info(f"Model Size before pruning:  {modelParamesOri}")
    logger.info(f"Model Size after pruning:   {modelParames}")
    logger.info(f"Model size remaining ratio: {remainingRatio}")
    logger.info(f"Model size sparsity:        {1-remainingRatio}")

    # load pca vocabulary
    pcaParamFileName = args.pcaVocabFileName
    pcaParamFileDir  = f'./pcaVocab/'
    pcaParamFilePath = os.path.join(pcaParamFileDir, pcaParamFileName) 
    logger.info(f"Loading pca vocab: {pcaParamFilePath}")
    with (open(pcaParamFilePath, "rb")) as openfile:
        pcaVocab = pickle.load(openfile)
    logger.info(f'Load pca vocab param pca matrix : {pcaVocab[VOCAB_PCA_PARAM][0].shape}')
    logger.info(f'Load pca vocab param pca mean   : {pcaVocab[VOCAB_PCA_PARAM][1].shape}')
    logger.info(f'Load pca vocab param lower dim embeddings : {pcaVocab[VOCAB_EMBEDDING_LOWER_DIM].shape}')
    logger.info(f'Load pca vocab token list : {len(pcaVocab[VOCAB_PCA_TOKEN_LIST])}')
    model.loadPCAVocab(pcaVocab)

    # setup evaluator
    testParam = {'batch_size': args.batch_size}
    tester = InTaskCoFiModelEvaluator(testParam, dataset)

    # evaluating it
    logger.info("Evaluating model ...")
    set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.
    acc, pre, rec, fsc = tester.evaluate(model, tok, None, logLevel='INFO')

    itemList = ["acc", "pre", "rec", "fsc"]
    logger.info("Evaluate statistics: ")
    printMean([acc, pre, rec, fsc], itemList, debugLevel=logging.INFO)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
