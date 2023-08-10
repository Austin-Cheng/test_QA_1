# -*- coding: utf-8 -*-
from typing import List

from app.constants.config import ENTITY_MODEL_FILE, ENT_TYPE_CN_EN_MAP
from app.utils.mention import Mention
from resources.ner.data_loader import NERDataset
from resources.ner.lilstm_crf import *
from torch.utils.data import DataLoader

from transformers import BertTokenizer
import torch


class ModelExtractor:
    """Extract entity mention from query by model."""
    def __init__(self):
        self.model = ''
        self.wordTOix = {}

    def load_model(self, path):
        """
        Load entity extractor model.
        Args:
            path: Model path.

        Returns:

        """
        if path is None:
            return
        checkpoint = torch.load(path)
        self.wordTOix = checkpoint['word_to_ix']
        self.model = BiLSTM_CRF(len(self.wordTOix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def extract(self, query) -> List[Mention]:
        """
        Extract entity from query by model.

        Args:
            query: Question text

        Returns: Entity mention list.

        """
        words = list(query)
        if len(words) == 0:
            return []
        precheck_sent = prepare_sequence(words, self.wordTOix)
        path_score, tag_seq, best_tag_score = self.model(precheck_sent)
        entities = extract_result(words, tag_seq, best_tag_score)   # 此处调用模型进行预测，模型输出实体，包含起止位置，类型，置信度 （text,type,start,end,score)
        mentions = convert2mention(entities)
        return mentions


def convert2mention(entities, default_confidence=1) -> List[Mention]:
    """
    把模型输出实体转化为mention格式
    标注标签为中文，词典规则标签为英文，若模型预测标签为中文，需转换为英文。转换对应map在app.constants.config.py

    Args:
        entities: 模型输出实体
        default_confidence: 若模型没有输出置信度，默认设置为某值

    Returns: mention列表

    """
    mentions = [Mention(''.join(ent[0]) if isinstance(ent[0], list) else ent[0],
                        ENT_TYPE_CN_EN_MAP[ent[1]], ent[2], ent[3],
                        confidence=ent[4] if len(ent)>=5 else default_confidence)
                for ent in entities]
    return mentions
