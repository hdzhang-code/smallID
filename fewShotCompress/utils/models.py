#coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random
from .transformers import AutoModelForMaskedLM
from utils.commonVar import *
from utils.Logger import logger
from torch.autograd import Variable
import math
from utils.transformers.modeling_utils import (apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils.transformers import AutoModelForMaskedLM, CoFiBertForSequenceClassification, AutoTokenizer, AutoConfig
from utils.transformers.models.bert.utils.cofi_utils import (prune_intermediate_layers)

from tqdm import tqdm
import pdb

class IntentBERT(nn.Module):
    def __init__(self, config):
        super(IntentBERT, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.featureDim = 768
        self.modelLayer = config['modelLayer']

        self.modelDir = config['modelDir']

        self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)
        self.dropout = nn.Dropout(0.1) # follow the default in bert model
        # self.word_embedding = nn.DataParallel(self.word_embedding)

        # load from Huggingface or from disk
        try:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(self.LMName)
        except:
            modelPath = os.path.join(self.modelDir, self.LMName)
            logger.info("Loading model from %s"%(modelPath))
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(modelPath)

        self.word_embedding.to(self.device)
        self.linearClsfier.to(self.device)

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output


    ##
    # @brief Add Spectral Decoupling, proposed in paper 'Gradient Starvation: A Learning Proclivity in Neural Networks'.
    #
    # @param logits
    # @param Y
    #
    # @return 
    def loss_ce_SD(self, logits, Y, sdWeight):
        sd = (logits**2).mean()
        loss = nn.CrossEntropyLoss()
        crossEntropy = loss(logits, Y)
        output = crossEntropy +  sdWeight * sd
        return output, crossEntropy, sd

    def SD(self, logits):
        sd = (logits**2).mean()
        return  sd

    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss

    def getUttEmbeddings(self, X):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        # CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = outputs.hidden_states[self.modelLayer][:,0]

        return CLSEmbedding


    def forwardEmbedding(self, X):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X)

        return CLSEmbedding

    def forwardEmbeddingDataLoader(self, dataLoader: torch.utils.data.DataLoader, detach=True):
        labelList = []
        embeddingList = []
        for batchID, batch in tqdm(enumerate(dataLoader)):
            Y, ids, types, masks = batch
            X = {'input_ids':ids.to(self.device),
                    'token_type_ids':types.to(self.device),
                    'attention_mask':masks.to(self.device)}

            # forward
            embeddings = self.forwardEmbedding(X)
            if detach:
                embeddings = embeddings.detach()
            labelList.append(Y)
            embeddingList.append(embeddings)

        labelListCat     = torch.cat(labelList)
        embeddingListCat = torch.cat(embeddingList)

        return labelListCat, embeddingListCat
    
    def forward(self, X, returnEmbedding=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X)

        # linear classifier
        logits = self.linearClsfier(CLSEmbedding)

        if returnEmbedding:
            return logits, CLSEmbedding
        else:
            return logits
    
    def fewShotPredict(self, supportX, supportY, queryX, clsFierName, mode='multi-class'):
        # calculate word embedding
        supportEmbedding = self.getUttEmbeddings(supportX)
        queryEmbedding   = self.getUttEmbeddings(queryX)

        # select clsfier
        support_features = supportEmbedding.cpu()
        query_features = queryEmbedding.cpu()
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_MULTI_LABEL:
            clf = MultiOutputClassifier(LogisticRegression(penalty='l2',
                                                           random_state=0,
                                                           C=1.0,
                                                           solver='liblinear',
                                                           max_iter=1000,
                                                           multi_class='ovr',
                                                           class_weight='balanced'))

            clf.fit(support_features, supportY)
        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        if mode == 'multi-class':
            query_pred = clf.predict(query_features)
        elif mode == 'entailment-utt':
            query_pred = clf.predict_proba(query_features)[:, 1]            
        elif mode == 'entailment-lab':
            query_pred = clf.predict_proba(query_features)[:, 1]

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        # pre-trained LM
        self.word_embedding.save_pretrained(path)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class LinearClsfier(nn.Module):
    def __init__(self, config):
        super(LinearClsfier, self).__init__()
        self.config = config  # debug
        self.device = config['device']
        self.clsNum = config['clsNumber']
        self.initializeValue = config['initializeValue']
        if 'loadModelPath' in config:
            self.loadModelPath     = config['loadModelPath']
        self.featureDim = 768
        if 'modelDir' in config:
            self.modelDir = config['modelDir']

        self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)
        if self.initializeValue is not None:
            self.linearClsfier.weight = torch.nn.Parameter(self.initializeValue)
            self.linearClsfier.bias   = torch.nn.Parameter(torch.zeros(self.linearClsfier.bias.shape))

        if hasattr(self, 'loadModelPath'):
            logger.info(f"Loading linear classifier from: {self.loadModelPath}")
            self.linearClsfier = torch.load(self.loadModelPath)
            self.linearClsfier.to(self.device)

    def forward(self, embeddings):
        # linear classifier
        logits = self.linearClsfier(embeddings)
        return logits

    def save(self, path):
        lcParamPath = f"{path}/{SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE}"
        torch.save(self.linearClsfier, lcParamPath)

    def loadFromDisk(self, path):
        lcParamPath = os.path.join(self.modelDir, path, SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE)
        logger.info(f"Loading linear classifier from: {lcParamPath}")
        self.linearClsfier = torch.load(lcParamPath)
        self.linearClsfier.to(self.device)


# This class is indeed the wrapper of the backbone. It contains three module: backbone, linear classifier and l0 module (masks).
class IntentBERTCoFi(nn.Module):
    def __init__(self, config, silence=False):
        super(IntentBERTCoFi, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.basemodel = config['basemodel']
        self.clsNum = config['clsNumber']
        if 'modelDir' in config:
            self.modelDir = config['modelDir']
        self.featureDim = 768

        # self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)   # for classification
        self.dropout = nn.Dropout(0.1) # follow the default in bert model
        # self.word_embedding = nn.DataParallel(self.word_embedding)

        # load from Huggingface or from disk
        try:
            self.backbone = CoFiBertForSequenceClassification.from_pretrained(self.LMName, silence=silence)
        except:
            modelPath = os.path.join(self.modelDir, self.LMName)
            logger.info("Loading model from %s"%(modelPath))
            self.backbone = CoFiBertForSequenceClassification.from_pretrained(modelPath, silence=silence)
            # lcParamPath = os.path.join(self.modelDir, self.LMName, SAVE_MODEL_LINEAR_CLASSIFER_PARAM_FILE)
            # logger.info(f"Loading linear classifier from: {lcParamPath}")
            # self.linearClsfier = torch.load(lcParamPath)


        # mask module, l0 module
        self.l0_module_droprate_init   = config['l0_module_droprate_init']
        self.l0_module_temperature     = config['l0_module_temperature']
        self.l0_module_target_sparsity = config['l0_module_target_sparsity']
        self.l0_module_pruning_type    = config['l0_module_pruning_type']
        l0Config = {}
        modelConfig = AutoConfig.from_pretrained(self.basemodel)
        self.l0_module = L0Module(config = modelConfig,
                             droprate_init = self.l0_module_droprate_init,
                             temperature = self.l0_module_temperature,
                             target_sparsity = self.l0_module_target_sparsity,
                             pruning_type = self.l0_module_pruning_type, silence=silence)

        self.backbone.to(self.device)
        self.l0_module.to(self.device)
        # self.linearClsfier.to(self.device)

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss

    def getUttEmbeddings(self, X, returnAllLayers=False):
        # BERT forward
        outputs = self.backbone(**X, output_hidden_states=True)

        if returnAllLayers:
            return outputs.hidden_states
        else:
            # extract [CLS] for utterance representation
            CLSEmbedding = outputs.hidden_states[-1][:,0]
            return CLSEmbedding


    def forwardEmbedding(self, X, returnAllLayers=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, returnAllLayers=returnAllLayers)

        return CLSEmbedding

    def forwardEmbeddingDataLoader(self, dataLoader: torch.utils.data.DataLoader, detach=True):
        labelList = []
        embeddingList = []
        for batchID, batch in tqdm(enumerate(dataLoader)):
            Y, ids, types, masks = batch
            X = {'input_ids':ids.to(self.device),
                    'token_type_ids':types.to(self.device),
                    'attention_mask':masks.to(self.device)}

            # forward
            embeddings = self.forwardEmbedding(X)
            if detach:
                embeddings = embeddings.detach()
            labelList.append(Y)
            embeddingList.append(embeddings)

        labelListCat     = torch.cat(labelList)
        embeddingListCat = torch.cat(embeddingList)

        return labelListCat, embeddingListCat
    
    def forward(self, X, returnEmbedding=False, returnAllLayers = False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, returnAllLayers=True)

        # linear classifier
        logits = self.linearClsfier(CLSEmbedding[-1][:,0])

        if returnEmbedding:
            return logits, CLSEmbedding
        else:
            return logits
    
    def fewShotPredict(self, supportX, supportY, queryX, clsFierName, mode='multi-class'):
        # calculate word embedding
        supportEmbedding = self.getUttEmbeddings(supportX)
        queryEmbedding   = self.getUttEmbeddings(queryX)

        # select clsfier
        support_features = supportEmbedding.cpu()
        query_features = queryEmbedding.cpu()
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_MULTI_LABEL:
            clf = MultiOutputClassifier(LogisticRegression(penalty='l2',
                                                           random_state=0,
                                                           C=1.0,
                                                           solver='liblinear',
                                                           max_iter=1000,
                                                           multi_class='ovr',
                                                           class_weight='balanced'))

            clf.fit(support_features, supportY)
        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        if mode == 'multi-class':
            query_pred = clf.predict(query_features)
        elif mode == 'entailment-utt':
            query_pred = clf.predict_proba(query_features)[:, 1]            
        elif mode == 'entailment-lab':
            query_pred = clf.predict_proba(query_features)[:, 1]

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)
    
    def normalize(self, x):
        norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
        out = x.div(norm)
        return out

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        # pre-trained LM
        self.backbone.save_pretrained(path)

    def _calculate_parameters(self):
        keys = ["embedding", "layer_transformation", "classifier", "pooler"]
        # keys = []
        return sum(p.numel() for n, p in self.backbone.named_parameters() if not any(key in n for key in keys))

    def calculate_parameters(self):
        modelParames = self._calculate_parameters()
        return modelParames

    # do pruning, call l0 module to get the mask, according to masks, do pruning, this is real pruning, i.e. some components of the backbone is removed.
    def prune_model_with_z(self):  
        # get deterministic masks.
        zs = self.l0_module.forward(training=False)
        if zs is None:
            return None, None
        bert = self.backbone.bert if hasattr(self.backbone, "bert") else model.roberta
       
        # prune head, according to head mask and head layer mask
        if "head_z" in zs:
            head_z = zs.get("head_z", None)
            head_layer_z = zs.get("head_layer_z", None)

            prune_heads = {}
            for layer in range(len(head_z)):
                head_z_layer = head_z[layer].cpu().squeeze().clone()
                if head_layer_z is not None:
                    head_z_layer *= head_layer_z[layer].cpu()
                index = torch.where(head_z_layer == 0)[0].tolist()
                prune_heads[layer] = index
            
                print(f"Layer {layer}, heads {' '.join([str(i) for i in index])} pruned.")
            self.backbone.prune_heads(prune_heads)

        kept_intermediate_dims = None
        if "intermediate_z" in zs:
            kept_intermediate_dims = {}
            intermediate_zs = zs["intermediate_z"]
            mlp_z = zs.get("mlp_z", None)
            for layer in range(len(intermediate_zs)):
                intermediate_z_layer = intermediate_zs[layer].squeeze()
                intermediate_z_layer = intermediate_z_layer.cpu().clone()
                if mlp_z is not None:
                    intermediate_z_layer *= mlp_z[layer].cpu()
                kept_intermediate_dims[layer] = intermediate_z_layer.nonzero().reshape(-1).tolist()

        def prune_layer_norm(layernorm, index):
            layernorm.weight = torch.nn.parameter.Parameter(
                layernorm.weight.index_select(0, index))
            layernorm.bias = torch.nn.parameter.Parameter(
                layernorm.bias.index_select(0, index))
            layernorm.normalized_shape = (len(index),)

        def prune_layer(layer, index, dim):
            layer = prune_linear_layer(layer, index, dim=dim)
            return layer

        if "hidden_z" in zs:
            hidden_zs = zs["hidden_z"]
            index = torch.LongTensor(hidden_zs.squeeze().nonzero().squeeze().tolist())
            index = index.to(self.device)
            
            bert.embeddings.word_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.word_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.word_embeddings.embedding_dim = index.shape[0]
            bert.embeddings.position_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.position_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.position_embeddings.embedding_dim = index.shape[0]
            bert.embeddings.token_type_embeddings.weight = torch.nn.parameter.Parameter(
                bert.embeddings.token_type_embeddings.weight.index_select(1, index).clone().detach())
            bert.embeddings.token_type_embeddings.embedding_dim = index.shape[0]
            prune_layer_norm(bert.embeddings.LayerNorm, index)

            for layer in range(0, 12):
                if bert.encoder.layer[layer].attention.self.query is not None:
                    bert.encoder.layer[layer].attention.self.query = \
                        prune_layer(bert.encoder.layer[layer].attention.self.query , index, dim=1)
                    bert.encoder.layer[layer].attention.self.key = \
                        prune_layer(bert.encoder.layer[layer].attention.self.key , index, dim=1)
                if bert.encoder.layer[layer].attention.self.value is not None:
                    bert.encoder.layer[layer].attention.self.value = \
                        prune_layer(bert.encoder.layer[layer].attention.self.value , index, dim=1)
                    bert.encoder.layer[layer].attention.output.dense = \
                        prune_layer(bert.encoder.layer[layer].attention.output.dense , index, dim=0)
                    prune_layer_norm(bert.encoder.layer[layer].attention.output.LayerNorm, index)
                if bert.encoder.layer[layer].intermediate.dense is not None:
                    bert.encoder.layer[layer].intermediate.dense = \
                        prune_layer( bert.encoder.layer[layer].intermediate.dense, index, dim=1)
                    bert.encoder.layer[layer].output.dense = \
                        prune_layer( bert.encoder.layer[layer].output.dense, index, dim=0)
                    prune_layer_norm(bert.encoder.layer[layer].output.LayerNorm, index)

            # prune linear classifier
            self.linearClsfier.linearClsfier = prune_linear_layer(self.linearClsfier.linearClsfier, index, dim=1)
            
            # accommodate for different models
            if hasattr(self.backbone, "classifier"):
                if hasattr(self.backbone.classifier, "dense"):
                    self.backbone.classifier.dense = prune_linear_layer(self.backbone.classifier.dense, index, dim=1)
            if hasattr(self.backbone, "cls"):
                if hasattr(self.backbone.cls, "dense"):
                    self.backbone.cls.dense = prune_linear_layer(self.backbone.classifier.dense, index, dim=1)
            if hasattr(bert.pooler, "dense"):
                bert.pooler.dense = prune_linear_layer(bert.pooler.dense, index, dim=1)
            if hasattr(self.backbone, "qa_outputs"):
                self.backbone.qa_outputs = prune_linear_layer(self.backbone.qa_outputs, index, dim=1)
            if getattr(self.backbone, "layer_transformation", None) is not None:
                self.backbone.layer_transformation = prune_linear_layer(self.backbone.layer_transformation, index, dim=1)
                print("layer transformation", self.backbone.layer_transformation.weight.shape)
            if getattr(self.backbone, "mha_layer_transformation", None) is not None:
                self.backbone.mha_layer_transformation = prune_linear_layer(self.backbone.mha_layer_transformation, index, dim=1)
                print("layer mha_layer_transformation", self.backbone.mha_layer_transformation.weight.shape)

        if kept_intermediate_dims is not None:
            prune_intermediate_layers(self.backbone, kept_intermediate_dims)

        for layer in range(0, 12):
            print("Layer:", layer)
            if bert.encoder.layer[layer].attention.self.query is not None:
                print("query:", bert.encoder.layer[layer].attention.self.query.weight.shape)
                print("key:", bert.encoder.layer[layer].attention.self.key.weight.shape)
            else:
                print("query:", None)
                print("key:", None)
            if bert.encoder.layer[layer].attention.self.value is not None:
                print("value:", bert.encoder.layer[layer].attention.self.value.weight.shape)
                print("output:", bert.encoder.layer[layer].attention.output.dense.weight.shape)
            else:
                print("value:", None)
                print("output:", None)
            if bert.encoder.layer[layer].intermediate.dense is not None:
                print("up:", bert.encoder.layer[layer].intermediate.dense.weight.shape)
                print("down:", bert.encoder.layer[layer].output.dense.weight.shape)
            else:
                print("up", None)
                print("down", None) 

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
class L0Module(nn.Module):
    def __init__(self,
                 config, 
                 droprate_init=0.5,
                 temperature=2./3.,
                 lagrangian_warmup=0,
                 start_sparsity=0.0,
                 target_sparsity=0.0,
                 pruning_type="structured_heads+structured_mlp+hidden+layer",
                 magical_number=0.8, # from Wang et al. 2020
                 silence=False
                 ):
        super(L0Module, self).__init__()
        self.silence = silence
        self.all_types = ["hidden_z", "intermediate_z", "mlp_z", "head_layer_z", "head_z"]
        self.pruning_type = pruning_type

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size 
        self.num_attention_heads = config.num_attention_heads
        self.mlp_num_per_layer = 1
        self.dim_per_head = self.hidden_size // self.num_attention_heads 
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size

        self.params_per_head_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * 4
        self.params_per_head =  self.params_per_head_layer // self.num_attention_heads
        

        self.params_per_mlp_layer = self.hidden_size * self.intermediate_size * 2 + self.hidden_size + self.hidden_size * 4
        self.params_per_intermediate_dim = self.params_per_mlp_layer // self.intermediate_size

        # we ignore the parameters in normalization layers (it takes a very small amount)
        self.full_model_size = (self.params_per_head_layer + self.params_per_mlp_layer) * self.num_hidden_layers
        self.prunable_model_size = 0 

        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        
        self.types = []
        self.z_logas = {}
        self.parameters_per_dim = {}
        self.sizes = {}
        self.shapes = {}

        self.hidden_loga = None
        self.hidden_type = None

        types = self.pruning_type.split("+")
        for type in types:
            if type != "layer":
                self.initialize_one_module(type)
        if "layer" in types:
            self.initialize_one_module("layer")

        self.magical_number = magical_number

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.0))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.0))

        self.lagrangian_warmup = lagrangian_warmup
        self.start_sparsity = start_sparsity
        self.target_sparsity = target_sparsity

        if not silence:
            logger.info("********** Initializing L0 Module **********") 
            for type in self.types:
                logger.info(f"***** {type} *****")
                logger.info(f"z.shape {self.z_logas[type].shape}")
                logger.info(f"size, {self.sizes[type]}")
            logger.info(f"prunable model size: {self.prunable_model_size}")

    def set_lagrangian_warmup_steps(self, lagrangian_warmup):
        self.lagrangian_warmup = lagrangian_warmup
        logger.info(f"Lagrangian warmup steps: {self.lagrangian_warmup}")

    def initialize_one_module(self, module_name):
        if module_name == "structured_mlp":
            self.initialize_structured_mlp()
        elif module_name == "structured_heads":
            self.initialize_structured_head()
        elif module_name == "hidden":
            self.initialize_hidden()
        elif module_name == "layer":
            self.initialize_whole_mlp()
            self.initialized_layer_structured_heads()
            
    def add_one_module(self, z_loga, type, parameter_per_dim, size, shape):
        self.types.append(type)
        self.z_logas[type] = z_loga
        self.parameters_per_dim[type] = parameter_per_dim
        self.sizes[type] = size
        self.shapes[type] = shape

    # randomly initialize log a of given shape
    def initialize_parameters(self, size, num_layer=None):
        if num_layer is not None:
            return Parameter(torch.Tensor(num_layer, size))
        else:
            return Parameter(torch.Tensor(size))

    def initialize_hidden(self):
        self.hidden_loga = self.initialize_parameters(self.hidden_size)
        self.add_one_module(self.hidden_loga, type="hidden", 
                            parameter_per_dim=self.hidden_size * 4 + self.hidden_size * 4 * 2,
                            size=self.hidden_size, shape=[self.hidden_size])
        self.reset_loga(self.hidden_loga, mean=10)
        if not self.silence:
            logger.info(f"Initialized hidden loga! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_head(self, add_prunable_model_size=True):
        self.head_loga = self.initialize_parameters(self.num_attention_heads, self.num_hidden_layers)
        self.reset_loga(self.head_loga, mean=10)  # reset loga so that all heads are kept
        self.add_one_module(self.head_loga, type="head", 
                            parameter_per_dim=self.params_per_head, size=self.num_attention_heads,
                            shape=[self.num_hidden_layers, 1, self.num_attention_heads, 1, 1])
        if add_prunable_model_size:
            self.prunable_model_size += self.params_per_head * self.num_hidden_layers * self.num_attention_heads
        if not self.silence:
            logger.info(f"Initialized structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialized_layer_structured_heads(self):
        n_layer = self.num_hidden_layers
        self.headlayer_loga = self.initialize_parameters(n_layer)
        self.reset_loga(self.headlayer_loga, mean=10)
        self.add_one_module(self.headlayer_loga, type="head_layer", 
                            parameter_per_dim=self.params_per_head * self.num_attention_heads, size=1,
                            shape=[n_layer])
        if not self.silence:
            logger.info(f"Initialized layerwise structured heads! Prunable_model_size = {self.prunable_model_size}")

    def initialize_structured_mlp(self):
        self.int_loga = self.initialize_parameters(self.intermediate_size, self.num_hidden_layers)

        self.add_one_module(self.int_loga, type="intermediate", 
                            parameter_per_dim=self.params_per_intermediate_dim, size=self.intermediate_size,
                            shape=[self.num_hidden_layers, 1, 1, self.intermediate_size])
        self.prunable_model_size += self.params_per_mlp_layer * self.num_hidden_layers
        # self.reset_loga(self.int_loga)
        self.reset_loga(self.int_loga, mean=10)
        if not self.silence:
            logger.info(f"Initialized structured mlp! Prunable_model_size = {self.prunable_model_size}")


    def initialize_whole_mlp(self):
        n_layer = self.num_hidden_layers
        self.intlayer_loga = self.initialize_parameters(n_layer)
        self.add_one_module(self.intlayer_loga, type="mlp", 
                            parameter_per_dim=self.params_per_mlp_layer, size=self.mlp_num_per_layer,
                            shape=[n_layer])
        self.reset_loga(self.intlayer_loga, mean=10)
        if not self.silence:
            logger.info(f"Initialized whole mlps! Prunable_model_size = {self.prunable_model_size}")


    def reset_loga(self, tensor, mean=None):
        if mean is None:
            mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        tensor.data.normal_(mean, 1e-2)

    def reset_qz_logas(self):
        for key in self.z_logas:
            if key in ["head_layer", "mlp", "head"]:
                self.reset_loga(self.z_logas[key], 10)
            else:
                self.reset_loga(self.z_logas[key])

    def constrain_parameters(self):
        def _constrain(tensor):
            tensor.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        for key in self.z_logas:
            _constrain(self.z_logas[key])

    def cdf_qz(self, x, loga):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, loga):
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def get_num_parameters_for_one(self, loga, parameter_size):
        return torch.sum(1 - self.cdf_qz(0, loga)) * parameter_size

    # transform the log a into the probability of being active. Two masks are considered here, head and head layer.
    def transform_scores_for_head(self):
        # assert "head" in self.types
        assert "head" in self.types

        if "head_layer" in self.types:
            all_head_score = 1 - self.cdf_qz(0, self.headlayer_loga) # cdf(0) denotes the probablity of the streched random variable to < 0, which means the corresponding mask is 0, or, is deactivated. Therefore, 1 - cdf(0) is the probabiliy of being active.
        else:
            all_head_score = None
        head_score = 1 - self.cdf_qz(0, self.head_loga) # 12 * 12
       
        if all_head_score is not None:
            all_head_score = all_head_score.view(-1, 1, 1) # 12 * 1 * 1
        head_score = head_score.unsqueeze(-1)   # 12 * 12 * 1
       
        return all_head_score, head_score

    def get_num_parameters_for_mlp(self):
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga) # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga) # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        num_parameters = torch.sum(intlayer_score * int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters

    # get expected l0 norm considering hidden dimension compression
    def get_num_parameters_and_constraint_for_hidden(self):
        num_parameters = 0
        constraint_loss = 0
       
        # head and head layer
        # 12 * 1 * 1
        # 12 * 12 * 1
        all_head_score, head_score = self.transform_scores_for_head()
        hidden_score = 1 - self.cdf_qz(0, self.hidden_loga) # 768

        if all_head_score is not None:
            head_score = (all_head_score * head_score).reshape(-1)
        else:
            head_score = head_score.reshape(-1)
        num_parameters += \
            torch.sum(torch.outer(hidden_score, head_score)) * self.parameters_per_dim["head"] / self.hidden_size

        # intermediate layer and intermediate dimension
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = (intlayer_score * int_score).reshape(-1)
        num_parameters += torch.sum(torch.outer(hidden_score, int_score)) * 2
        return num_parameters, constraint_loss


    def get_num_parameters_and_constraint(self):
        num_parameters = 0
        constraint_loss = 0

        # head and head layer
        all_head_score, head_score = self.transform_scores_for_head()
        
        head_score = head_score * all_head_score
        num_parameters += torch.sum(head_score) * self.parameters_per_dim["head"]

        # intermediate layer
        intlayer_score = 1 - self.cdf_qz(0, self.intlayer_loga)  # 12
        int_score = 1 - self.cdf_qz(0, self.int_loga)  # 12 * 3072
        intlayer_score = intlayer_score.unsqueeze(-1)

        int_score = int_score * intlayer_score
        num_parameters += torch.sum(int_score) * self.parameters_per_dim["intermediate"]
        return num_parameters, constraint_loss


    def get_target_sparsity(self, pruned_steps):
        target_sparsity = (self.target_sparsity - self.start_sparsity) * min(1, pruned_steps / self.lagrangian_warmup) + self.start_sparsity
        return target_sparsity

    def getExpectedModelSparsity(self):
        # calculate the expected l0 norm
        if "hidden" in self.types:
            expected_size, constraint = self.get_num_parameters_and_constraint_for_hidden()
        else:
            expected_size, constraint = self.get_num_parameters_and_constraint()

        # calculate the expected sparsity
        expected_sparsity = 1 - expected_size.detach().cpu().item() / self.prunable_model_size
        return expected_sparsity


    def lagrangian_regularization(self, pruned_steps):
        # calculate the expected l0 norm
        target_sparsity = self.target_sparsity
        if "hidden" in self.types:
            expected_size, constraint = self.get_num_parameters_and_constraint_for_hidden()
        else:
            expected_size, constraint = self.get_num_parameters_and_constraint()

        # calculate the loss: it is roughly the difference between the expected l0 norm and target l0 norm.
        expected_sparsity = 1 - expected_size / self.prunable_model_size
        if self.lagrangian_warmup > 0:
            target_sparsity = self.get_target_sparsity(pruned_steps)
        lagrangian_loss = (
                self.lambda_1 * (expected_sparsity - target_sparsity)
                + self.lambda_2 * (expected_sparsity - target_sparsity) ** 2
        )
        lagrangian_loss += constraint
        return lagrangian_loss, expected_sparsity, target_sparsity

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = torch.FloatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    # during training
    def _sample_z(self, loga):
        eps = self.get_eps(torch.FloatTensor(*loga.shape)).to(loga.device)
        z = self.quantile_concrete(eps, loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    # during inference, the expected zeros, k, in the mask is calculated and then the top k mask values are set 0, before the mask is returned
    def _deterministic_z(self, size, loga):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        expected_num_nonzeros = torch.sum(1 - self.cdf_qz(0, loga))   # expected l0 norm
        expected_num_zeros = size - expected_num_nonzeros.item()
        try:
            num_zeros = round(expected_num_zeros)
        except:
            pdb.set_trace()
        soft_mask = torch.sigmoid(loga / self.temperature * self.magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    def get_z_from_zs(self, zs):
        numpified_zs = {} 
        for type in self.all_types:
            name = z.split("_")[0]
            z = zs.get(type, np.ones(self.shapes[name]))
            if torch.is_tensor(z): 
                new_z = z.squeeze().cpu().numpy() > 0
            numpified_zs[type] = new_z
        return numpified_zs

    def calculate_model_size(self, zs):
        numpified_zs = self.get_z_from_zs(zs)
        
        hidden_z = numpified_zs["hidden"]
        intermediate_z = numpified_zs["intermediate"]
        mlp_z = numpified_zs["mlp"]
        head_z = numpified_zs["head"]
        head_layer_z = numpified_zs["head_layer"]

        remaining_hidden_dims = hidden_z.sum().item()
        remaining_intermediate_nums = intermediate_z.reshape(self.num_hidden_layers, self.intermediate_size).sum(-1).tolist()
        remaining_head_nums = head_z.reshape(self.num_hidden_layers, self.num_attention_heads).sum(-1).tolist()

        head_nums = np.outer((head_z * head_layer_z).reshape(-1), hidden_z).sum().item()
        intermediate_nums = np.outer((intermediate_z * mlp_z).reshape(-1), hidden_z).sum().item()

        remaining_model_size = head_nums * self.params_per_head_layer + intermediate_nums * self.params_per_head
        pruned_model_size = self.prunable_model_size - remaining_model_size

        results = {}
        # Not multiplied with each other
        results["head_layers"] = head_layer_z
        results["mlp_layers"] = mlp_z
        results["hidden_dims"] = remaining_hidden_dims
        results["intermediate_dims"] = remaining_intermediate_nums
        results["head_nums"] = remaining_head_nums
        results["pruned_params"] = pruned_model_size
        results["remaining_params"] = remaining_model_size
        results["pruned_model_sparsity"] = pruned_model_size / self.prunable_model_size

        # logger.info(f"remaining_head_layers: {head_layer_z}")
        # logger.info(f"remaining_mlp_layers: {mlp_z}")
        # logger.info(f"remaining_hidden_dims: {remaining_hidden_dims}")
        # logger.info(f"remaining_intermediate_nums: {remaining_intermediate_nums}")
        # logger.info(f"remaining_head_nums: {remaining_head_nums}")
        # logger.info(f"pruned_model_size: {pruned_model_size}")
        # logger.info(f"remaining_model_size: {remaining_model_size}")


    def forward(self, training=True,):
        zs = {f"{type}_z": [] for type in self.types}

        if training:
            for i, type in enumerate(self.types):
                loga = self.z_logas[type]
                z = self._sample_z(loga)
                zs[f"{type}_z"] = z.reshape(self.shapes[type])
        else:
            for i, type in enumerate(self.types):
                if type != "hidden": # hidden is not a per layer sample
                    loga_all_layers = self.z_logas[type]
                    for layer in range(len(loga_all_layers)):
                        loga = loga_all_layers[layer]
                        size = self.sizes[type]
                        z = self._deterministic_z(size, loga)
                        zs[f"{type}_z"].append(z.reshape(self.shapes[type][1:]))
                else:
                    z = self._deterministic_z(self.sizes[type], self.hidden_loga)
                    zs[f"{type}_z"] = z
            for type in zs:
                if type != "hidden_z":
                    zs[type] = torch.stack(zs[type])
        return zs 
