import utils.transformers as transformers
from scipy.stats import norm
from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
from utils.models import IntentBERT, LinearClsfier
from utils.models import IntentBERTCoFi, LinearClsfier
import time
import torch
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from utils.transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy as cp
import wandb
from sklearn.decomposition import PCA
import pdb

##
# @brief  base class of trainer
class TrainerBase():
    def __init__(self, wandb, wandbProj, wandbConfig, wandbRunName):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4

        # wandb 
        self.wandb = wandb
        self.wandbProjName = wandbProj
        self.wandbConfig = wandbConfig
        self.runName = wandbRunName
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class TransferTrainer(TrainerBase):
    def __init__(self,
            trainingParam:dict,
            optimizer,
            dataset:IntentDataset,
            valEvaluator: EvaluatorBase):
        super(TransferTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']

        self.dataset       = dataset
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator

        self.batchMonitor = trainingParam["batchMonitor"]

    def train(self, model, tokenizer):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        # evaluate before training
        valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer)
        logger.info('---- Before training ----')
        logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)

        # construct training data loader
        labTensorData = self.dataset.trainPart.generateTorchDataset(tokenizer)
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandb.run.name=(self.runName)

        earlyStopFlag = False
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            batchTrLossMLMSum = 0.0
            timeEpochStart    = time.time()

            timeMonitorWindowStart = time.time()
            batchNum = len(dataloader)
            for batchID, batch in enumerate(dataloader):
                model.train()
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}

                # forward
                logits, embeddings = model(X, returnEmbedding=True)
                # loss
                lossSP = model.loss_ce(logits, Y.to(model.device))

                # lossTOT = lossSP
                lossTOT = lossSP

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                acc = accuracy_score(YTensor, predictResult)

                if (batchID % self.batchMonitor) == 0:
                    # self.batchMonitor number batch training done, collect data
                    model.eval()
                    valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer)

                    # statistics
                    monitorWindowDurationTrain = self.round(time.time() - timeMonitorWindowStart)

                    # display current epoch's info
                    logger.info("---- epoch: %d/%d, batch: %d/%d, monitor window time %f ----", epoch, self.epoch, batchID, batchNum, self.round(monitorWindowDurationTrain))
                    logger.info("TrainLoss %f", lossTOT.item())
                    logger.info("valAcc %f, valPre %f, valRec %f , valFsc %f", valAcc, valPre, valRec, valFsc)
                    if self.wandb:
                        wandb.log({'trainLoss': lossTOT.item(), \
                                'trainAcc': acc, \
                                'valAcc': valAcc, \
                                'lossCE': lossSP, \
                                })

                    # time
                    timeMonitorWindowStart = time.time()
                    durationOverallTrain += monitorWindowDurationTrain

                    # early stop
                    if not self.validation:
                        valAcc = -1
                    if (valAcc >= valBestAcc):   # better validation result
                        print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                        valBestAcc = valAcc
                        accumulateStep = 0

                        # cache current model, used for evaluation later
                        self.bestModelStateDict = copy.deepcopy(model.state_dict())
                    else:
                        accumulateStep += 1
                        if accumulateStep > self.patience/2:
                            print('[INFO] accumulateStep: ', accumulateStep)
                            if accumulateStep == self.patience:  # early stop
                                logger.info('Early stop.')
                                logger.debug("Overall training time %f", durationOverallTrain)
                                logger.debug("best_val_acc: %f", valBestAcc)
                                earlyStopFlag = True
                                break

            if earlyStopFlag:
                break

        if self.wandb:
            run.finish()

        logger.debug('All %d epochs are finished', self.epoch)
        logger.debug("Overall training time %f", durationOverallTrain)
        logger.info("best_val_acc: %f", valBestAcc)

##
# @brief TransferTrainer used to do transfer-training. The training is performed in a supervised manner. All available data is used fo training. By contrast, meta-training is performed by tasks. 
class TransferTrainer(TrainerBase):
    def __init__(self,
            trainingParam:dict,
            optimizer,
            dataset:IntentDataset,
            valEvaluator: EvaluatorBase):
        super(TransferTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']

        self.dataset       = dataset
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator

        self.batchMonitor = trainingParam["batchMonitor"]

    def train(self, model, tokenizer):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        # evaluate before training
        valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer)
        logger.info('---- Before training ----')
        logger.info("ValAcc %f, Val pre %f, Val rec %f , Val Fsc %f", valAcc, valPre, valRec, valFsc)

        # construct training data loader
        labTensorData = self.dataset.trainPart.generateTorchDataset(tokenizer)
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandb.run.name=(self.runName)

        earlyStopFlag = False
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            batchTrLossMLMSum = 0.0
            timeEpochStart    = time.time()

            timeMonitorWindowStart = time.time()
            batchNum = len(dataloader)
            for batchID, batch in enumerate(dataloader):
                model.train()
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}

                # forward
                logits, embeddings = model(X, returnEmbedding=True)
                # loss
                lossSP = model.loss_ce(logits, Y.to(model.device))

                # lossTOT = lossSP
                lossTOT = lossSP

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                acc = accuracy_score(YTensor, predictResult)

                if (batchID % self.batchMonitor) == 0:
                    # self.batchMonitor number batch training done, collect data
                    model.eval()
                    valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer)

                    # statistics
                    monitorWindowDurationTrain = self.round(time.time() - timeMonitorWindowStart)

                    # display current epoch's info
                    logger.info("---- epoch: %d/%d, batch: %d/%d, monitor window time %f ----", epoch, self.epoch, batchID, batchNum, self.round(monitorWindowDurationTrain))
                    logger.info("TrainLoss %f", lossTOT.item())
                    logger.info("valAcc %f, valPre %f, valRec %f , valFsc %f", valAcc, valPre, valRec, valFsc)
                    if self.wandb:
                        wandb.log({'trainLoss': lossTOT.item(), \
                                'trainAcc': acc, \
                                'valAcc': valAcc, \
                                'lossCE': lossSP, \
                                })

                    # time
                    timeMonitorWindowStart = time.time()
                    durationOverallTrain += monitorWindowDurationTrain

                    # early stop
                    if not self.validation:
                        valAcc = -1
                    if (valAcc >= valBestAcc):   # better validation result
                        print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                        valBestAcc = valAcc
                        accumulateStep = 0

                        # cache current model, used for evaluation later
                        self.bestModelStateDict = copy.deepcopy(model.state_dict())
                    else:
                        accumulateStep += 1
                        if accumulateStep > self.patience/2:
                            print('[INFO] accumulateStep: ', accumulateStep)
                            if accumulateStep == self.patience:  # early stop
                                logger.info('Early stop.')
                                logger.debug("Overall training time %f", durationOverallTrain)
                                logger.debug("best_val_acc: %f", valBestAcc)
                                earlyStopFlag = True
                                break

            if earlyStopFlag:
                break

        if self.wandb:
            run.finish()

        logger.debug('All %d epochs are finished', self.epoch)
        logger.debug("Overall training time %f", durationOverallTrain)
        logger.info("best_val_acc: %f", valBestAcc)

##
# @brief FewShotTuneTrainer used to do in-task fine-tuning on the K few-shot data for few-shot intent detection. We abandon the concept of task in meta-learning. K-shot data is sampled from the whole dataset, imitating supervised learning. In other words, we finetune the model on a small portion of training data.
class FewShotTuneTrainer(TrainerBase):
    def __init__(self, trainingParam, dataset: IntentDataset):
        super(FewShotTuneTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.shot  = trainingParam['shot']
        self.batch_size = trainingParam['batch_size']
        self.seed = trainingParam['seed']
        self.inTaskEpoch = trainingParam['inTaskEpoch']
        self.monitorTestPerform = trainingParam['monitorTestPerform']


        self.dataset      = dataset
        self.lrBackbone          = trainingParam['lrBackbone']
        self.lrClsfier           = trainingParam['lrClsfier']
        self.weight_decay = trainingParam['weight_decay']
        self.sdWeight     = trainingParam['sdWeight']

        self.epochMonitorWindow = trainingParam['epochMonitorWindow']

    def train(self, model, tokenizer, dataset, dataloader, logLevel='DEBUG'):
        # initialize linear classifier with the prototype of data
        logger.info("Initializing linear classifier ...")
        lc = self.initLinearClassifierRandom(model, dataset)

        # fine-tune the model with K-shot data
        logger.info("Fine-tuning the model with K-shot data ...")
        self.fineTuneKshot(model, lc, dataset, dataloader, tokenizer)

        return model, lc

    def initLinearClassifierRandom(self, model, dataset):
        # initialize linear classifier
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None}
        lc = LinearClsfier(lcConfig)

        return lc

    def initLinearClassifier(self, model, dataset, dataloader):
        # calculate proto for classifier initialization
        model.eval()
        Y, embeddings = model.forwardEmbeddingDataLoader(dataloader)

        Y = Y.detach().cpu()
        embeddings = embeddings.detach().cpu()
        Y2EmbeddingList = {}
        for y, embeddings in zip(Y, embeddings):
            yValue = y.item()
            if yValue not in Y2EmbeddingList:
                Y2EmbeddingList[yValue] = []
            Y2EmbeddingList[yValue].append(embeddings)
        protoList = []
        for y in sorted(Y2EmbeddingList):
            proto = torch.stack(Y2EmbeddingList[y]).mean(0)
            protoList.append(proto)
        protoTensor = torch.stack(protoList)

        # initialize linear classifier
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': protoTensor}
        lc = LinearClsfier(lcConfig)

        return lc

    def fineTuneKshot(self, model, lc, dataset, dataloader, tokenizer):
        # finetune
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandbRunName = self.runName
            wandb.run.name=(wandbRunName)

        paramList = [{'params': model.parameters(), 'lr': self.lrBackbone}, \
                {'params': lc.parameters(), 'lr': self.lrClsfier}]
        optimizer = optim.AdamW(paramList, weight_decay=self.weight_decay)
        t_total = len(dataloader) * self.inTaskEpoch
        warmup_steps = round(t_total/20)
        logger.info(f"Learning rate scheduler: warmup_steps={warmup_steps}, t_total={t_total}.")
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        for epoch in range(self.inTaskEpoch):
            model.train()
            lc.train()

            batchLossList = []
            batchCEList = []
            batchSDList = []
            trainAccList = []
            for batchID, batch in enumerate(dataloader):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                # loss function
                lossCE = model.loss_ce(logits, Y.to(model.device))
                lossSD = model.SD(logits)
                # loss, lossCE, lossSD = model.loss_ce_SD(logits, Y.to(model.device), self.sdWeight)
                loss = lossCE + self.sdWeight * lossSD

                batchLossList.append(loss.item())
                batchCEList.append(lossCE.item())
                batchSDList.append(lossSD.item())
                optimizer.zero_grad()
                loss.backward()
                # logger.info(f"Fine-tuning inside task: loss = {loss.item()}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(lc.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logger.info("Learning rate:")
                logger.info([group['lr'] for group in optimizer.param_groups])

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                trainAcc = accuracy_score(YTensor, predictResult)
                trainAccList.append(trainAcc)

            # this epoch is done
            avrgLoss     = sum(batchLossList) / len(batchLossList)
            avrgLossCE   = sum(batchCEList) / len(batchCEList)
            avrgLossSD   = sum(batchSDList) / len(batchSDList)
            avrgTrainAcc = sum(trainAccList) / len(trainAccList)

            # validation on test partition
            logger.info("Monitoring performance on test partition ...")
            if self.monitorTestPerform:
                acc, pre, rec, fsc = self.evaluateOnTestPartition(model, lc, dataset, tokenizer)
            else:
                acc = -1
                pre = -1
                rec = -1
                fsc = -1
            logger.info(f"In-task fine-tuning epoch {epoch}, averLoss = {avrgLoss}, avrgLossCE={avrgLossCE}, avrgLossSD={avrgLossSD}, tranAcc={avrgTrainAcc}, testPartAcc={acc}.")

            # log in wandb
            if self.wandb:
                if epoch % self.epochMonitorWindow == 0:
                    wandb.log({'avrgLoss': avrgLoss, \
                            'avrgLossCE': avrgLossCE, \
                            'avrgLossSD': avrgLossSD, \
                            'avrgTrainAcc': avrgTrainAcc, \
                            'epoch': epoch, \
                            'testPartAcc': acc, \
                            })

        if self.wandb:
            run.finish()

    ##
    # @brief 
    #
    # @param model
    # @param lc linear classifier
    # @param tokenizer
    #
    # @return 
    def evaluateOnTestPartition(self, model, lc, dataset, tokenizer):
        # evaluate on test partition
        model.eval()
        lc.eval()
        with torch.no_grad():
            # loop test partition to predict the label
            tensorDatasetTest = dataset.testPart.generateTorchDataset(tokenizer)
            dataloaderTest = DataLoader(tensorDatasetTest, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            YTruthList = []
            predResultList = []
            for batchID, batch in tqdm(enumerate(dataloaderTest)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predResult = np.argmax(logits, 1)

                YTruthList.extend(Y.tolist())
                predResultList.extend(predResult.tolist())
        # calculate acc
        acc = accuracy_score(YTruthList, predResultList)   # acc
        performDetail = precision_recall_fscore_support(YTruthList, predResultList, average='macro', warn_for=tuple())

        return acc, performDetail[0], performDetail[1], performDetail[2]

##
# @brief CompressTrainer is used to compress model.
class CompressTrainer(TrainerBase):
    def __init__(self, trainingParam, dataset: IntentDataset):
        super(CompressTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.shot  = trainingParam['shot']
        self.batch_size = trainingParam['batch_size']
        self.seed = trainingParam['seed']
        self.inTaskEpoch = trainingParam['inTaskEpoch']
        self.monitorTestPerform = trainingParam['monitorTestPerform']


        self.alpha = trainingParam['alpha']
        self.dataset      = dataset
        self.lrBackbone   = trainingParam['lrBackbone']
        self.lrClsfier    = trainingParam['lrClsfier']
        self.weight_decay = trainingParam['weight_decay']
        self.sdWeight     = trainingParam['sdWeight']
        self.distill_temperature     = trainingParam['distill_temperature']

        self.kd_criterion = nn.KLDivLoss(reduction="none")

        self.epochMonitorWindow = trainingParam['epochMonitorWindow']

    ##
    # @brief 
    #
    # @param model  teacher model
    # @param modelStudent
    # @param lc
    # @param tokenizer
    # @param dataset
    # @param dataloaderUnlabel
    # @param logLevel
    #
    # @return 
    def train(self, model, modelStudent, teacherLC, tokenizer, dataset, dataloaderUnlabel, logLevel='DEBUG',  fewDataDataLoader=None):
        # initialize linear classifier with the prototype of data
        logger.info("Initializing linear classifier ...")
        studentLC = self.initLinearClassifierRandom(model, dataset)

        # perform knowledge distillation on unlabeled data
        logger.info("Knowledge distillation ...")
        self.knowledgeDistill(model, modelStudent, studentLC, teacherLC, dataset, dataloaderUnlabel, tokenizer, fewDataDataLoader)

        return modelStudent, studentLC

    def initLinearClassifierRandom(self, model, dataset):
        # initialize linear classifier
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None}
        lc = LinearClsfier(lcConfig)

        return lc


    ##
    # @brief return soft target, the logit output of the teacher model, and also the weight of each data point which is calculated according to the distance between it and the proto of the few data. 
    #
    # @param model
    # @param lc
    # @param dataloader
    # @param fewUttList
    # @param fewLabList
    #
    # @return 
    def getSoftTarget(self, model, lc, dataloader, fewDataDataLoader):
        model.eval()
        # calculate few data proto
        logger.info(f"Calculating few data proto ...")
        with torch.no_grad():
            lab2embedding = {}
            for batchID, batch in tqdm(enumerate(fewDataDataLoader)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)

                labList = Y.tolist()
                if not isinstance(labList, list):
                    labList = [labList]
                for lab, embedding in zip(labList, batchEmbedding):
                    if lab not in lab2embedding:
                        lab2embedding[lab] = []
                    lab2embedding[lab].append(embedding)

            lab2proto = {}
            for lab in lab2embedding:
                lab2proto[lab] = torch.mean(torch.stack(lab2embedding[lab]), dim=0)

        # calculate soft target
        dataDict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'Y': [], 'logits': [], 'weights':[]}
        input_ids_list = []
        token_type_ids_list = []
        attention_mask_list = []
        Y_list = []
        logits_list = []
        dotSim_list = []
        with torch.no_grad():
            # calculate embeddings: label -> list of embeddings
            Y2Embeddings = {}
            embeddingList = []
            logger.info(f"Calculating  variance...")
            for batchID, batch in tqdm(enumerate(dataloader)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                input_ids_list.extend(ids.cpu())
                token_type_ids_list.extend(types.cpu())
                attention_mask_list.extend(masks.cpu())
                Y_list.extend(Y.cpu())
                logits_list.extend(logits.cpu())

                # similarity/dist
                labList = Y.tolist()
                if not isinstance(labList, list):
                    labList = [labList]
                protos  = torch.stack([lab2proto[lab] for lab in labList])
                dotSim  = torch.bmm(protos.unsqueeze(1), batchEmbedding.unsqueeze(2)).squeeze()
                dotSim  = torch.clip(dotSim, min=0)
                dotSim_list.extend([sim.item() for sim in dotSim])

                # build up Y2Embeddings
                yList = Y.tolist()
                for y, embedding in zip(yList, batchEmbedding):
                    if y not in Y2Embeddings:
                        Y2Embeddings[y] = []
                    Y2Embeddings[y].append(embedding)

            # calculate standard deviation
            logger.info(f"Calculating standard deviation ...")
            Y2SimalrityList = []
            y2std = {}
            for y in Y2Embeddings:
                embeddings = Y2Embeddings[y]
                embeddingsStack = torch.stack(embeddings)
                proto = lab2proto[y]
                similarity = proto @ embeddingsStack.T
                similarity = torch.clip(similarity, min=0)
                dist = -similarity + max(similarity)
                y2std[y] = torch.std(dist).item()

            # calculate weights
            logger.info(f"Calculating sample weights ...")
            y2indexList = {}
            Y_item_list = [y.item() for y in Y_list]
            for ind, y in enumerate(Y_item_list):
                if y not in y2indexList:
                    y2indexList[y] = []
                y2indexList[y].append(ind)
            weights_list = [-1] * len(Y_list)
            for y in y2indexList:
                indexList = y2indexList[y]
                dotSim_list_currentLab = torch.Tensor([dotSim_list[index] for index in indexList])
                distCurrtentLab = -dotSim_list_currentLab + max(dotSim_list_currentLab)
                # distCurrtentLab = distCurrtentLab.numpy()

                loc = 0
                std = y2std[y]
                std = std * self.alpha
                weights = norm.pdf(distCurrtentLab, loc, std)
                # weights = F.normalize(torch.from_numpy(weights.squeeze()), dim=0)
                weights = weights / weights.sum()

                for index, weight in zip(indexList, weights):
                    weights_list[index] = weight

            dataDict['input_ids'] = input_ids_list
            dataDict['token_type_ids'] = token_type_ids_list
            dataDict['attention_mask'] = attention_mask_list
            dataDict['Y'] = Y_list
            dataDict['logits'] = logits_list
            dataDict['weights'] = torch.Tensor(weights_list)

            if not len(set([len(dataDict[key]) for key in dataDict])) == 1:
                logger.error(f"Inconsistent size in dataDict.")
                return 

        return dataDict
            
    def knowledgeDistill(self, model, studentModel, studentLC, teacherLC, dataset, dataloader, tokenizer, fewDataDataLoader):
        # finetune
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandbRunName = self.runName
            wandb.run.name=(wandbRunName)

        # get soft target from teacher network
        logger.info(f"Generating teacher soft target ...")
        targetSoftLabel = self.getSoftTarget(model, teacherLC, dataloader, fewDataDataLoader)
        tensorDatasetWithLogits = TensorDataset(torch.stack(targetSoftLabel['Y']),   \
                torch.stack(targetSoftLabel['input_ids']), \
                torch.stack(targetSoftLabel['token_type_ids']), \
                torch.stack(targetSoftLabel['attention_mask']), \
                torch.stack(targetSoftLabel['logits']), \
                targetSoftLabel['weights'], \
                )
        tensorDataloaderWithLogits = DataLoader(tensorDatasetWithLogits, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # prepare optimizer
        # paramList = [{'params': model.parameters(), 'lr': self.lrBackbone}, \
        #         {'params': lc.parameters(), 'lr': self.lrClsfier}]
        paramList = [{'params': studentModel.parameters(), 'lr': self.lrBackbone}, \
                {'params': studentLC.parameters(), 'lr': self.lrClsfier}]
        optimizer = optim.AdamW(paramList, weight_decay=self.weight_decay)
        t_total = len(dataloader) * self.inTaskEpoch
        warmup_steps = round(t_total/20)
        logger.info(f"Learning rate scheduler: warmup_steps={warmup_steps}, t_total={t_total}.")
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        for epoch in range(self.inTaskEpoch):
            studentModel.train()
            studentLC.train()

            batchLossList = []
            batchCEList = []
            batchSDList = []
            trainAccList = []
            for batchID, batch in enumerate(tensorDataloaderWithLogits):
                # batch data
                Y, ids, types, masks, logitsTeacher, weights = batch
                X = {'input_ids':ids.to(studentModel.device),
                        'token_type_ids':types.to(studentModel.device),
                        'attention_mask':masks.to(studentModel.device)}
                batchEmbedding = studentModel.forwardEmbedding(X)
                logits = studentLC(batchEmbedding)

                # loss function
                logitsTeacher = logitsTeacher.to(studentModel.device)

                if True:
                    losses = self.kd_criterion(F.log_softmax(logits / self.distill_temperature, 1), F.softmax(logitsTeacher / self.distill_temperature, 1))
                    losses  = losses.mean(1)
                    weights = weights.to(studentModel.device)
                    # loss = losses * F.softmax(weights/0.5)
                    # loss = losses * F.softmax(weights/0.1)
                    # loss = losses * F.softmax(weights/2.0)
                    # loss = losses * F.softmax(weights/5.0)
                    # weights = F.normalize(torch.from_numpy(weights.squeeze()), dim=0)
                    # weights = F.normalize(weights.squeeze(), dim=0)
                    weights = weights/weights.sum()
                    loss = losses * weights
                    # loss = loss.mean()
                    loss = loss.sum()

                batchLossList.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                # logger.info(f"Fine-tuning inside task: loss = {loss.item()}")
                torch.nn.utils.clip_grad_norm_(studentModel.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(studentLC.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logger.info("Learning rate:")
                logger.info([group['lr'] for group in optimizer.param_groups])

            # this epoch is done
            avrgLoss = sum(batchLossList) / len(batchLossList)

            # validation on test partition
            logger.info("Monitoring performance on test partition ...")
            if self.monitorTestPerform:
                acc, pre, rec, fsc = self.evaluateOnTestPartition(studentModel, studentLC, dataset, tokenizer)
            else:
                acc = -1
                pre = -1
                rec = -1
                fsc = -1
            logger.info(f"Compression epoch {epoch}, averLoss = {avrgLoss}, testPartAcc={acc}.")

            # log in wandb
            if self.wandb:
                if epoch % self.epochMonitorWindow == 0:
                    wandb.log({'avrgLoss': avrgLoss, \
                            'epoch': epoch, \
                            'testPartAcc': acc, \
                            })

        if self.wandb:
            run.finish()

    ##
    # @brief 
    #
    # @param model
    # @param lc linear classifier
    # @param tokenizer
    #
    # @return 
    def evaluateOnTestPartition(self, model, lc, dataset, tokenizer):
        # evaluate on test partition
        model.eval()
        lc.eval()
        with torch.no_grad():
            # loop test partition to predict the label
            tensorDatasetTest = dataset.testPart.generateTorchDataset(tokenizer)
            dataloaderTest = DataLoader(tensorDatasetTest, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            YTruthList = []
            predResultList = []
            for batchID, batch in tqdm(enumerate(dataloaderTest)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()
                logits = logits.numpy()
                predResult = np.argmax(logits, 1)

                YTruthList.extend(Y.tolist())
                predResultList.extend(predResult.tolist())
        # calculate acc
        acc = accuracy_score(YTruthList, predResultList)   # acc
        performDetail = precision_recall_fscore_support(YTruthList, predResultList, average='macro', warn_for=tuple())

        return acc, performDetail[0], performDetail[1], performDetail[2]

##
# @brief CoFiTrainer is used to prune the model following CoFi.
class CoFiFewShotTrainer(TrainerBase):
    def __init__(self,
            trainingParam:dict,
            dataset:IntentDataset,
            validator
            ):
        super(CoFiFewShotTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.shot = trainingParam['shot']
        self.batch_size  = trainingParam['batch_size']
        self.dataset       = dataset
        self.lr           = trainingParam['lr']
        self.weight_decay = trainingParam['weight_decay']

        self.prepruning_finetune_epochs = trainingParam['prepruning_finetune_epochs']
        self.lagrangian_warmup_epochs = trainingParam['lagrangian_warmup_epochs']
        self.pruning_epochs = trainingParam['pruning_epochs']
        self.distill_temp = 2.0
        self.distill_loss_alpha = 0.7
        self.distill_ce_loss_alpha = 0.3

        self.validator = validator

    def setWandbRunName(self, name):
        self.runName = name

    def fineTuneModel(self, model, dataloader):
        logger.info("Initializing linear classifier ...")
        model.eval()
        Y, embeddings = model.forwardEmbeddingDataLoader(dataloader)
        # calculate proto for classifier initialization
        Y = Y.detach().cpu()
        embeddings = embeddings.detach().cpu()
        Y2EmbeddingList = {}
        for y, embeddings in zip(Y, embeddings):
            yValue = y.item()
            if yValue not in Y2EmbeddingList:
                Y2EmbeddingList[yValue] = []
            Y2EmbeddingList[yValue].append(embeddings)
        protoList = []
        for y in sorted(Y2EmbeddingList):
            proto = torch.stack(Y2EmbeddingList[y]).mean(0)
            protoList.append(proto)
        protoTensor = torch.stack(protoList)

        # finetune
        model.train()
        # initialize linear classifier
        lcConfig = {'device': model.device, 'clsNumber': self.dataset.getLabNum(), 'initializeValue': protoTensor}
        model.linearClsfier = LinearClsfier(lcConfig)
        model.linearClsfier.to(model.device)
        optimizer = optim.AdamW(list(model.backbone.parameters()) + list(model.linearClsfier.parameters()), lr=self.lr, weight_decay=self.weight_decay)   # optimizer backbone and linear classifier
        patience = 5
        lowLossEpoch = 0
        for epoch in range(self.prepruning_finetune_epochs):
            batchLossList = []
            for batchID, batch in enumerate(dataloader):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                logits = model.forward(X)
                loss = model.loss_ce(logits, Y.to(model.device))
                batchLossList.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(model.linearClsfier.parameters(), 1.0)
                optimizer.step()
            avrgLoss = sum(batchLossList) / len(batchLossList)
            if avrgLoss < 0.01:
                lowLossEpoch = lowLossEpoch + 1
            else:
                lowLossEpoch = 0
            logger.info(f"Pre-pruning fine-tuning epoch {epoch}, averLoss = {avrgLoss}, lowLossE = {lowLossEpoch}/{patience}")
            if lowLossEpoch >= patience:
                break

    def pruneModel(self, model, dataloader, tok):
        # wandb
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandb.run.name=(self.runName)

        # prepare stuffs for pruning
        # prepare l0 module
        num_update_steps_per_epoch = len(dataloader)
        lagrangian_warmup_steps = self.lagrangian_warmup_epochs * num_update_steps_per_epoch
        model.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
        # prepare optimizer
        t_total = self.pruning_epochs * num_update_steps_per_epoch
        optSchConfig = {}
        # opt params to optimize main paramters: backbone + linear classifier
        optSchConfig['weight_decay'] =  0.0
        optSchConfig['learning_rate'] = 2e-5
        optSchConfig['adam_beta1'] = 0.9
        optSchConfig['adam_beta2'] = 0.999
        optSchConfig['adam_epsilon'] = 1e-8
        optSchConfig['reg_learning_rate'] =  0.01
        optSchConfig['warmup_steps'] = 0
        optimizer, l0_optimizer, lagrangian_optimizer, lr_scheduler = self.create_optimizer_and_scheduler(optSchConfig, model, t_total)
        model.train()
        teacher_model = cp.deepcopy(model)
        teacher_model.eval()
        globalPruneStep = 0
        for epoch in range(self.pruning_epochs):
            batchLossList = []
            for batchID, batch in enumerate(dataloader):
                # prepare inputs
                # batch data
                Y, ids, types, masks = batch
                inputs = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                # generate mask
                zs = model.l0_module.forward(training=True)
                self.fill_inputs_with_zs(zs, inputs)
                distill_loss = None
                distill_ce_loss = None
                if teacher_model is not None:
                    with torch.no_grad():
                        # only retain inputs of certain keys
                        teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                               "output_attentions", "output_hidden_states", "return_dict"]
                        teacher_inputs = {key: inputs[key]
                                          for key in teacher_inputs_keys if key in inputs}
                        teacher_logits, teacher_outputs = teacher_model.forward(teacher_inputs, returnEmbedding=True, returnAllLayers=True)
                    student_logits, student_outputs = model.forward(inputs, returnEmbedding=True, returnAllLayers=True)
                    zs = {key: inputs[key] for key in inputs if "_z" in inputs}
                    distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(teacher_outputs, student_outputs, zs, model, teacher_logits, student_logits)
                else:
                    loss = self.compute_loss(model, inputs)

                lagrangian_loss = None
                lagrangian_loss, _, _ = model.l0_module.lagrangian_regularization(globalPruneStep)
                loss += lagrangian_loss

                loss.backward()
                # logger.info(f"loss: {loss.detach().item()}")
                # logger.info(f"lagrangian_loss: {lagrangian_loss.detach().item()}")
                # logger.info(f"distill_layer_loss: {distill_loss.detach().item()}")
                # logger.info(f"distill_ce_loss: {distill_ce_loss.detach().item()}")
                # logger.info(f"expected sparsity: {model.l0_module.getExpectedModelSparsity()}")

                # optimize
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                l0_optimizer.step()
                lagrangian_optimizer.step()
                lr_scheduler.step()
                model.l0_module.constrain_parameters()
                # zero gradients
                optimizer.zero_grad()
                l0_optimizer.zero_grad()
                lagrangian_optimizer.zero_grad()

                # total_flos += self.floating_point_ops(inputs)  # it is used to calculate FLOPS, temporarily comment it off.
                globalPruneStep = globalPruneStep + 1

            # logger.info(f"Pre-pruning fine-tuning epoch {epoch}, averLoss = {avrgLoss}, lowLossE = {lowLossEpoch}/{patience}")
            logger.info(f"Pruning epoch {epoch}/{self.pruning_epochs}.")
            logger.info(f"loss: {loss.detach().item()}")
            logger.info(f"lagrangian_loss: {lagrangian_loss.detach().item()}")
            logger.info(f"distill_layer_loss: {distill_loss.detach().item()}")
            logger.info(f"distill_ce_loss: {distill_ce_loss.detach().item()}")
            logger.info(f"expected sparsity: {model.l0_module.getExpectedModelSparsity()}")

            if epoch % 100 == 0:
                logger.info(f"Checking model performance ...")
                acc = self.checkModelPerformance(model, tok, dataloader)
                if self.wandb:
                    wandb.log({'epoch': epoch, \
                            'loss': loss.detach().item(), \
                            'lagrangian_loss': lagrangian_loss.detach().item(), \
                            'distill_layer_loss': distill_loss.detach().item(), \
                            'distill_ce_loss': distill_ce_loss.detach().item(), \
                            'expected sparsity': model.l0_module.getExpectedModelSparsity(), \
                            'val acc': acc, \
                            })

        if self.wandb:
            run.finish()

        return 0


    ##
    # @brief copy current model, prune it
    #
    # @param model
    # @param tok
    # @param kShotDataLoader
    #
    # @return 
    def checkModelPerformance(self, model, tok, kShotDataLoader):
        # copy the model
        modelConfig = {}
        device = model.device
        modelConfig['device'] = device
        modelConfig['clsNumber'] = 90
        modelConfig['LMName'] = 'bert-base-uncased'
        modelConfig['basemodel'] = 'bert-base-uncased'
        modelConfig['l0_module_droprate_init'] = model.l0_module_droprate_init
        modelConfig['l0_module_temperature'] = model.l0_module_temperature
        modelConfig['l0_module_target_sparsity'] = model.l0_module_target_sparsity
        modelConfig['l0_module_pruning_type'] = model.l0_module_pruning_type
        modelCopy = IntentBERTCoFi(modelConfig, silence=True)
        modelCopy.linearClsfier = LinearClsfier(model.linearClsfier.config)
        modelCopy.load_state_dict(model.state_dict())
        # prune 
        modelCopy.prune_model_with_z()
        # evaluate
        acc, pre, rec, F1 = self.validator.evaluate(modelCopy, tok, kShotDataLoader)
        logger.info(f'acc: {acc}')
        logger.info(f'pre: {pre}')
        logger.info(f'rec: {rec}')
        logger.info(f'F1:  {F1}')
        return acc
    
    ##
    # @brief randomly sample K-shot data from dataset
    #
    # @return dataloader containing the K-shot data
    def sampleKShotData(self, tokenizer):
        # sample K-shot in the training partition
        tensorDataset = self.dataset.trainPart.randomSliceTorchDataset(self.shot, tokenizer, returnUttListLabList=False)
        dataloader = DataLoader(tensorDataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        return dataloader


    ##
    # @brief prune the model
    #
    # @param model
    # @param tokenizer
    # @param dataloader K-shot data, used for pruning
    #
    # @return 
    def train(self, model, tokenizer, dataloader):
        # calculate model size before pruning
        modelParamesOri = model.calculate_parameters()

        # step 1. Pruning
        logger.info(f"Pruning model ...")
        self.pruneModel(model, dataloader, tokenizer)
        logger.info(f"Pruning model ... done.")

        # copy the model
        modelConfig = {}
        device = model.device
        modelConfig['device'] = device
        modelConfig['clsNumber'] = 90
        modelConfig['LMName'] = 'bert-base-uncased'
        modelConfig['basemodel'] = 'bert-base-uncased'
        modelConfig['l0_module_droprate_init'] = model.l0_module_droprate_init
        modelConfig['l0_module_temperature'] = model.l0_module_temperature
        modelConfig['l0_module_target_sparsity'] = model.l0_module_target_sparsity
        modelConfig['l0_module_pruning_type'] = model.l0_module_pruning_type
        modelCopy = IntentBERTCoFi(modelConfig, silence=True)
        modelCopy.linearClsfier = LinearClsfier(model.linearClsfier.config)
        modelCopy.load_state_dict(model.state_dict())
        modelWithTrainedMasks = modelCopy
        # count model size
        model.prune_model_with_z()
        modelParames = model.calculate_parameters()
        remainingRatio = modelParames / modelParamesOri
        logger.info(f"Model Size before pruning:  {modelParamesOri}")
        logger.info(f"Model Size after pruning:   {modelParames}")
        logger.info(f"Model size remaining ratio: {remainingRatio}")
        logger.info(f"Model size sparsity:        {1-remainingRatio}")
        return modelWithTrainedMasks, model

    def create_optimizer_and_scheduler(self, config, model, num_training_steps: int):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                        f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        no_decay = ["bias", "LayerNorm.weight"]
        freeze_keywords = ["embeddings"]

        # backbone + classifier
        main_model_params = [
                {
                    "params": [p for n, p in list(model.backbone.named_parameters()) + list(model.linearClsfier.named_parameters()) if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": config['weight_decay'],
                    "lr": config['learning_rate']
                    },
                {
                    "params": [p for n, p in list(model.backbone.named_parameters()) + list(model.linearClsfier.named_parameters()) if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": config['learning_rate']
                    },
                ]
        log_params(main_model_params, "main params: backbone + linear classifier")
        optimizer = optim.AdamW(
                main_model_params,
                betas=(config['adam_beta1'], config['adam_beta2']),
                eps=config['adam_epsilon'],
                )
        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=num_training_steps)

        # l0 module paramters
        l0_params = [{
            "params": [p for n, p in model.l0_module.named_parameters() if "lambda" not in n],
            "weight_decay": 0.0,
            "lr":config['reg_learning_rate']
            }]
        log_params(l0_params, "l0 reg params")
        l0_optimizer = optim.AdamW(l0_params,
                betas=(config['adam_beta1'],
                    config['adam_beta2']),
                eps=config['adam_epsilon'])

        # lagrangian parameters
        lagrangian_params = [{
            "params": [p for n, p in model.l0_module.named_parameters() if "lambda" in n],
            "weight_decay": 0.0,
            "lr": -config['reg_learning_rate']
            }]
        log_params(lagrangian_params, "l0 reg lagrangian params")
        lagrangian_optimizer = optim.AdamW(lagrangian_params,
                betas=(config['adam_beta1'],
                    config['adam_beta2']),
                eps=config['adam_epsilon'])

        return optimizer, l0_optimizer, lagrangian_optimizer, lr_scheduler

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs, model, teacher_logits, student_logits):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs, model) # intermediate representation distillation
        distill_loss = layer_loss

        # ce_distill_loss = F.kl_div(   # soft target distillation
        #     input=F.log_softmax(
        #         student_outputs[1] / self.distill_temp, dim=-1),
        #     target=F.softmax(
        #         teacher_outputs[1] / self.distill_temp, dim=-1),
        #     reduction="batchmean") * (self.distill_temp ** 2)
        ce_distill_loss = F.kl_div(   # soft target distillation
            input=F.log_softmax(
                student_logits / self.distill_temp, dim=-1),
            target=F.softmax(
                teacher_logits / self.distill_temp, dim=-1),
            reduction="batchmean") * (self.distill_temp ** 2)

        loss = self.distill_loss_alpha * distill_loss + \
            self.distill_ce_loss_alpha * ce_distill_loss

        return distill_loss, ce_distill_loss, loss

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs, model):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        do_layer_distill = True
        if do_layer_distill:
            mlp_z = None
            head_layer_z = None
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            # teacher_layer_output = teacher_outputs[2][1:]
            teacher_layer_output = teacher_outputs[1:]
            # student_layer_output = student_outputs[2][1:]
            student_layer_output = student_outputs[1:]

            # distilliting existing layers
            layer_distill_version = 3
            if layer_distill_version == 2:
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    s_layer_o = model.layer_transformation(s_layer_o)
                    l = mse_loss(t_layer_o, s_layer_o)
                    if mlp_z[layer_num] > 0:
                        layer_loss += l

            # distilling layers with a minimal distance
            elif layer_distill_version > 2:
                l = []
                specified_teacher_layers = [2, 5, 8, 11]
                transformed_s_layer_o = [model.backbone.layer_transformation(
                    s_layer_o) for s_layer_o in student_layer_output]
                specified_teacher_layer_reps = [
                    teacher_layer_output[i] for i in specified_teacher_layers]

                device = transformed_s_layer_o[0].device
                for t_layer_o in specified_teacher_layer_reps:
                    for i, s_layer_o in enumerate(transformed_s_layer_o):
                        l.append(mse_loss(t_layer_o, s_layer_o))
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps), len(student_layer_output))

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0

                layer_loss = 0
                # no ordering restriction specified
                if layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                # added the ordering restriction
                elif layer_distill_version == 4:
                    last_aligned_layer = 12
                    alignment = []
                    for search_index in range(3, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(4).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum()
            return layer_loss
        else:
            return 0

class PCATrainer(TrainerBase):
    def __init__(self, trainingParam, dataset: IntentDataset):
        super(PCATrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.shot  = trainingParam['shot']
        self.batch_size = trainingParam['batch_size']
        self.seed = trainingParam['seed']

        self.dataset      = dataset

    def calculateVocabEmbeddings(self, model, tokenizer, vocabList):
        # vocab to tokens
        tokenList = []
        logger.info(f'Calculating embeddings for vcoab size: {len(vocabList)}.')
        for word in vocabList:
            token = tokenizer.convert_tokens_to_ids(word)
            tokenList.append(token)
        logger.info(f'Token list size: {len(tokenList)}.')

        embeddingList = []
        for token in tokenList:
            embedding = model.fromTokenToEmbedding(token) 
            embeddingList.append(embedding)
        return embeddingList, tokenList

    def train(self, model, tokenizer, vocabList,  pcaDim = -1, logLevel='DEBUG'):
        model.eval()
        with torch.no_grad():
            # loop all tokens in vocabulary
            embeddingList, tokenList = self.calculateVocabEmbeddings(model, tokenizer, vocabList)
            logger.info(f"Calculate embeddings for vocablist: {len(embeddingList)}")

            # calculate PCA transofrmation
            embeddingNumpy = torch.stack(embeddingList).numpy()
            pca = PCA(n_components = pcaDim)
            pca.fit(embeddingNumpy)
            matrixPCA = pca.components_   # components * orignal x dim
            embeddingNumpy_mean = np.mean(embeddingNumpy, axis=0)
            embeddingNumpyLowerDim = np.transpose(np.matmul(matrixPCA, np.transpose(embeddingNumpy - embeddingNumpy_mean)))
            pass

        pcaParam = (matrixPCA, embeddingNumpy_mean)

        return pcaParam, embeddingNumpyLowerDim, tokenList
