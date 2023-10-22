import transformers
from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
from utils.models import IntentBERT, LinearClsfier
import time
import torch
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
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


        self.dataset      = dataset
        self.lrBackbone   = trainingParam['lrBackbone']
        self.lrClsfier    = trainingParam['lrClsfier']
        self.weight_decay = trainingParam['weight_decay']
        self.sdWeight     = trainingParam['sdWeight']
        self.distill_temperature     = trainingParam['distill_temperature']

        self.kd_criterion = nn.KLDivLoss(reduction="batchmean")

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
    def train(self, model, modelStudent, lc, tokenizer, dataset, dataloaderUnlabel, logLevel='DEBUG'):
        # initialize linear classifier with the prototype of data
        logger.info("Initializing linear classifier ...")
        lc = self.initLinearClassifierRandom(model, dataset)

        # perform knowledge distillation on unlabeled data
        logger.info("Knowledge distillation ...")
        self.knowledgeDistill(model, modelStudent, lc, dataset, dataloaderUnlabel, tokenizer)

        return modelStudent, lc

    def initLinearClassifierRandom(self, model, dataset):
        # initialize linear classifier
        lcConfig = {'device': model.device, 'clsNumber': dataset.getLabNum(), 'initializeValue': None}
        lc = LinearClsfier(lcConfig)

        return lc

    def getSoftTarget(self, model, lc, dataloader):
        dataDict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'Y': [], 'logits': []}
        model.eval()
        with torch.no_grad():
            for batchID, batch in tqdm(enumerate(dataloader)):
                # batch data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}
                batchEmbedding = model.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                dataDict['input_ids'].extend(ids.cpu())
                dataDict['token_type_ids'].extend(types.cpu())
                dataDict['attention_mask'].extend(masks.cpu())
                dataDict['Y'].extend(Y.cpu())
                dataDict['logits'].extend(logits.cpu())

        return dataDict
            

    def knowledgeDistill(self, model, studentModel, lc, dataset, dataloader, tokenizer):
        # finetune
        if self.wandb:
            run = wandb.init(project=self.wandbProjName, reinit=True)
            wandb.config.update(self.wandbConfig)
            wandbRunName = self.runName
            wandb.run.name=(wandbRunName)

        # get soft target from teacher network
        logger.info(f"Generating teacher soft target ...")
        targetSoftLabel = self.getSoftTarget(model, lc, dataloader)
        tensorDatasetWithLogits = TensorDataset(torch.stack(targetSoftLabel['Y']),   \
                torch.stack(targetSoftLabel['input_ids']), \
                torch.stack(targetSoftLabel['token_type_ids']), \
                torch.stack(targetSoftLabel['attention_mask']), \
                torch.stack(targetSoftLabel['logits']), \
                )
        tensorDataloaderWithLogits = DataLoader(tensorDatasetWithLogits, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # prepare optimizer
        # paramList = [{'params': model.parameters(), 'lr': self.lrBackbone}, \
        #         {'params': lc.parameters(), 'lr': self.lrClsfier}]
        paramList = [{'params': studentModel.parameters(), 'lr': self.lrBackbone}, \
                {'params': lc.parameters(), 'lr': self.lrClsfier}]
        optimizer = optim.AdamW(paramList, weight_decay=self.weight_decay)
        t_total = len(dataloader) * self.inTaskEpoch
        warmup_steps = round(t_total/20)
        logger.info(f"Learning rate scheduler: warmup_steps={warmup_steps}, t_total={t_total}.")
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        for epoch in range(self.inTaskEpoch):
            studentModel.train()
            lc.train()

            batchLossList = []
            batchCEList = []
            batchSDList = []
            trainAccList = []
            for batchID, batch in enumerate(tensorDataloaderWithLogits):
                # batch data
                Y, ids, types, masks, logitsTeacher = batch
                X = {'input_ids':ids.to(studentModel.device),
                        'token_type_ids':types.to(studentModel.device),
                        'attention_mask':masks.to(studentModel.device)}
                batchEmbedding = studentModel.forwardEmbedding(X)
                logits = lc(batchEmbedding)

                # loss function
                logitsTeacher = logitsTeacher.to(studentModel.device)
                loss = self.kd_criterion(F.log_softmax(logits / self.distill_temperature, 1), F.softmax(logitsTeacher / self.distill_temperature, 1))

                batchLossList.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                # logger.info(f"Fine-tuning inside task: loss = {loss.item()}")
                torch.nn.utils.clip_grad_norm_(studentModel.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(lc.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                logger.info("Learning rate:")
                logger.info([group['lr'] for group in optimizer.param_groups])

            # this epoch is done
            avrgLoss = sum(batchLossList) / len(batchLossList)

            # validation on test partition
            logger.info("Monitoring performance on test partition ...")
            if self.monitorTestPerform:
                acc, pre, rec, fsc = self.evaluateOnTestPartition(studentModel, lc, dataset, tokenizer)
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
