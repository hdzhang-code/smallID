# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.
# This file is coded based on train_matchingNet.py.
# coding=utf-8
import torch
import argparse
import time
from transformers import AutoTokenizer

from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Evaluator import FewShotEvaluator, InTaskFineTuneEvaluator
from utils.Trainer import FewShotTuneTrainer
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
from torch.utils.data import DataLoader
import json
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
    parser.add_argument('--modelLayer', default=12, type=int)

    # ==== dataset ====
    parser.add_argument('--testDataset', help="Dataset names included in this experiment. For example:'OOS'.")
    parser.add_argument('--testDomain', help='Test domain names and separated by comma.')
    parser.add_argument('--genFewDataOnly', action="store_true")
    
    # ==== evaluation ====
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--inTaskEpoch', type=int, default=2)
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--lrBackbone', type=float, default=2e-5)
    parser.add_argument('--lrClsfier', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)
    parser.add_argument('--sdWeight', default=0.0, type=float)
    parser.add_argument('--epochMonitorWindow', type=int, default=1)
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--wandb', help='use wandb or not', action="store_true")
    parser.add_argument('--wandbProj', help='wandb project name')
    parser.add_argument('--saveModel', action='store_true')
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--monitorTestPerform', action='store_true')

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
    logger.info(f"Loading data from {args.testDataset}")
    dataset = IntentDataset(args.testDataset, mode=DATASET_MODE_PARTITION_MERGE_INTO_TEST)
    dataset = dataset.selectByDomains(splitName(args.testDomain))
    dataset.tokenize(tok)

    # ======= prepare model ======
    # initialize model
    logger.info("----- IntentBERT initializing -----")
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = 90
    modelConfig['LMName'] = args.LMName
    modelConfig['modelLayer'] = 12
    model = IntentBERT(modelConfig)

    # setup evaluator
    trainingParam = {"shot":args.shot,  \
            'batch_size': args.batch_size,  \
            'lrBackbone': args.lrBackbone, \
            'lrClsfier': args.lrClsfier, \
            'weight_decay': args.weightDecay,  \
            'seed':args.seed,  \
            'inTaskEpoch':args.inTaskEpoch,  \
            'sdWeight': args.sdWeight,
            "wandb": args.wandb, \
            "wandbProj":args.wandbProj, \
            "wandbRunName": "lrBackbone%.5f-lrCls%.5f-Te%s-%s-seed%d-st%s"%(args.lrBackbone, args.lrClsfier, args.testDataset, args.LMName, args.seed, args.shot), \
            "wandbConfig": {},  \
            "epochMonitorWindow" : args.epochMonitorWindow, \
            "monitorTestPerform": args.monitorTestPerform
    }
    trainer = FewShotTuneTrainer(trainingParam, dataset)

    set_seed(args.seed)   # reset seed so that the seed can control the following random data sample process.

    # sample K-shot in the training partition
    tensorDataset, uttList, labList = dataset.trainPart.randomSliceTorchDataset(args.shot, tok, returnLabList = True)
    dataloader = DataLoader(tensorDataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    logger.info(f"Sliced {args.shot} data from dataset. K-shot data size is {len(tensorDataset)}.")

    if args.genFewDataOnly:
        # Serializing json
        dictionary = {}
        dictionary[TRANSFER_FEW_SHOT_UTT_LIST] = uttList
        dictionary[TRANSFER_FEW_SHOT_LAB_LIST] = labList
        json_object = json.dumps(dictionary)

        # decide the save name
        save_path = os.path.join(FEW_SHOT_DATA_DIR, f"fewShotFineTune_{args.testDataset}_seed{args.seed}_shot{args.shot}.json")
        # save
        if not os.path.exists(FEW_SHOT_DATA_DIR):
            os.makedirs(FEW_SHOT_DATA_DIR)
        logger.info("Saving sampled few-shot data into file: %s", save_path)
        with open(save_path, "w") as outfile:
            outfile.write(json_object)
        

    exit(0)   # this python script is used only to generate few shot, save them out for later usage.

    # fine-tune
    logger.info("On K few-shot data, fine-tuning model ...")
    model, lc = trainer.train(model, tok, dataset, dataloader, logLevel='INFO')

    # save model in disk
    if args.saveModel:
        # decide the save name
        save_path = os.path.join(SAVE_PATH, args.saveName)
        # save
        logger.info("Saving model.pth into folder: %s", save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save(save_path)
        lc.save(save_path)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
