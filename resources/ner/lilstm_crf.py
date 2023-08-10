import os
import json
import logging
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from numpy import *

from app.constants.config import RESOURCES_DIR

device = torch.device('cuda:0')

# 为CPU中设置种子，生成随机数
torch.manual_seed(1)


# 得到最大值的索引
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix.get(w, 0) for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
# 等同于torch.log(torch.sum(torch.exp(vec)))，防止e的指数导致计算机上溢
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # 转移矩阵，transaction[i][j]表示从label_j转移到label_i的概率，虽然是随机生成的，但是后面会迭代更新
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # 设置任何标签都不可能转移到开始标签。设置结束标签不可能转移到其他任何标签
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # 随机初始化lstm的输入（h_0，c_0）
        self.hidden = self.init_hidden()

    # 随机生成输入的h_0,c_0
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg_new(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tagset_size], -10000.)
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)  # +1
            aa = gamar_r_l + t_r1_k + self.transitions
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    # 求所有可能路径得分之和
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # 输入：发射矩阵，实际就是LSTM的输出————sentence的每个word经LSTM后，对应于每个label的得分
        # 输出：所有可能路径得分之和/归一化因子/配分函数/Z(x)
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # 包装到一个变量里以便自动反向传播
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # 当前层这一点的发射得分要与上一层所有点的得分相加，为了用加快运算，将其扩充为相同维度的矩阵
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # 前一层5个previous_tags到当前层当前tag_i的transition scors
                trans_score = self.transitions[next_tag].view(1, -1)
                # 前一层所有点的总得分 + 前一节点标签转移到当前结点标签的得分（边得分） + 当前点的发射得分
                next_tag_var = forward_var + trans_score + emit_score
                # 求和，实现w_(t-1)到w_t的推进
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            # 保存的是当前层所有点的得分
            forward_var = torch.cat(alphas_t).view(1, -1)
        # 最后将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # 输入：id化的自然语言序列
        # 输出：序列中每个字符的Emission Score
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm模型的输出矩阵维度为（seq_len，batch，num_direction*hidden_dim）
        # 所以此时lstm_out的维度为（11,1,4）
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # 把batch维度去掉，以便接入全连接层
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # 用一个全连接层将其转换为（seq_len，tag_size）维度，才能生成最后的Emission Score
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        # 输入：feats——emission scores；tag——真实序列标注，以此确定转移矩阵中选择哪条路径
        # 输出：真是路径得分
        score = torch.zeros(1)
        # 将START_TAG的标签3拼接到tag序列最前面
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        # 路径得分等于：前一点标签转移到当前点标签的得分 + 当前点的发射得分
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        # 最后加上STOP_TAG标签的转移得分，其发射得分为0，可以忽略
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        # 预测路径得分，维特比解码，输出得分与路径值
        backpointers = []
        step_var = []
        # Initialize the viterbi variables in log space
        # B:0  I:1  O:2  START_TAG:3  STOP_TAG:4
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        # 维特比解码的开始：一个START_TAG，得分设置为0，其他标签的得分可设置比0小很多的数
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var表示当前这个字被标注为各个标签的得分（概率）
        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        # 遍历每个字，过程中取出这个字的发射得分
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            # 遍历每个标签，计算当前字被标注为当前标签的得分
            for next_tag in range(self.tagset_size):
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # forward_var保存的是之前的最优路径的值，然后加上转移到当前标签的得分，
                # 得到当前字被标注为当前标签的得分（概率）
                next_tag_var = forward_var + self.transitions[next_tag]
                # 找出上一个字中的哪个标签转移到当前next_tag标签的概率最大，并把它保存下载
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                # 把最大的得分也保存下来
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # 然后加上各个节点的发射分数，形成新一层的得分
            # cat用于将list中的多个tensor变量拼接成一个tensor
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # 得到了从上一字的标签转移到当前字的每个标签的最优路径
            # bptrs_t有5个元素
            backpointers.append(bptrs_t)
            step_var.append(forward_var.tolist())

        # 其他标签到结束标签的转移概率
        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        best_tag_id_org = best_tag_id
        # 最终的最优路径得分
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        best_tag_score = []
        # best_path_score.append(path_score.item())
        for step in reversed(step_var):
            step_score = step[0][best_tag_id_org]
            best_tag_score.append(step_score)
        # Pop off the start tag (we dont want to return that to the caller)
        # 无需返回最开始的start标签
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        # 把从后向前的路径正过来
        best_path.reverse()
        return path_score, best_path, best_tag_score

    # 损失函数
    def neg_log_likelihood(self, sentence, tags):
        # len(s)*5
        feats = self._get_lstm_features(sentence)
        # 规范化因子 | 配分函数 | 所有路径的得分之和
        forward_score = self._forward_alg_new(feats)
        # 正确路径得分
        gold_score = self._score_sentence(feats, tags)
        # 已取反
        # 原本CRF是要最大化gold_score - forward_score，但深度学习一般都最小化损失函数，所以给该式子取反
        return forward_score - gold_score

    # 实际上是模型的预测函数，用来得到一个最佳的路径以及路径得分
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 解码过程，维特比解码选择最大概率的标注路径

        # 先放入BiLstm模型中得到它的发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # 然后使用维特比解码得到最佳路径
        path_score, tag_seq, best_tag_score = self._viterbi_decode(lstm_feats)
        return path_score, tag_seq, best_tag_score


START_TAG = "<START>"
STOP_TAG = "<STOP>"

labels = ['异常情况','异常类别','异常原因','软件','软件版本','功能','命令','命令描述','命令参数',
          '命令参数描述','解决方案','软件升级流程','实体类型_命令','实体类型_命令参数','实体类型_命令描述','实体类型_命令参数描述',
          '实体类型_异常原因','实体类型_异常现象','实体类型_类别','实体类型_解决方案','实体类型_软件升级流程','实体类型_功能','实体类型_软件',
          '实体类型_软件版本','实体类型_软件升级服务','非具体异常现象']

tag_to_ix = {
    "O": 0,
    "B-异常情况": 1,
    "B-异常类别": 2,
    "B-异常原因": 3,
    "B-软件": 4,
    "B-软件版本": 5,
    "B-功能": 6,
    "B-命令": 7,
    "B-命令描述": 8,
    "B-命令参数": 9,
    "B-命令参数描述": 10,
    "B-解决方案": 11,
    "B-软件升级流程": 12,
    "B-实体类型_命令": 13,
    "B-实体类型_命令参数": 14,
    "B-实体类型_命令描述": 15,
    "B-实体类型_命令参数描述": 16,
    "B-实体类型_异常原因": 17,
    "B-实体类型_异常现象": 18,
    "B-实体类型_类别": 19,
    "B-实体类型_解决方案": 20,
    "B-实体类型_软件升级流程": 21,
    "B-实体类型_功能": 22,
    "B-实体类型_软件": 23,
    "B-实体类型_软件版本": 24,
    "B-实体类型_软件升级服务": 25,
    "B-非具体异常现象": 26,
    "I-异常情况": 27,
    "I-异常类别": 28,
    "I-异常原因": 29,
    "I-软件": 30,
    "I-软件版本": 31,
    "I-功能": 32,
    "I-命令": 33,
    "I-命令描述": 34,
    "I-命令参数": 35,
    "I-命令参数描述": 36,
    "I-解决方案": 37,
    "I-软件升级流程": 38,
    "I-实体类型_命令": 39,
    "I-实体类型_命令参数": 40,
    "I-实体类型_命令描述": 41,
    "I-实体类型_命令参数描述": 42,
    "I-实体类型_异常原因": 43,
    "I-实体类型_异常现象": 44,
    "I-实体类型_类别": 45,
    "I-实体类型_解决方案": 46,
    "I-实体类型_软件升级流程": 47,
    "I-实体类型_功能": 48,
    "I-实体类型_软件": 49,
    "I-实体类型_软件版本": 50,
    "I-实体类型_软件升级服务": 51,
    "I-非具体异常现象": 52,
    "S-异常情况": 53,
    "S-异常类别": 54,
    "S-异常原因": 55,
    "S-软件": 56,
    "S-软件版本": 57,
    "S-功能": 58,
    "S-命令": 59,
    "S-命令描述": 60,
    "S-命令参数": 61,
    "S-命令参数描述": 62,
    "S-解决方案": 63,
    "S-软件升级流程": 64,
    "S-实体类型_命令": 65,
    "S-实体类型_命令参数": 66,
    "S-实体类型_命令描述": 67,
    "S-实体类型_命令参数描述": 68,
    "S-实体类型_异常原因": 69,
    "S-实体类型_异常现象": 70,
    "S-实体类型_类别": 71,
    "S-实体类型_解决方案": 72,
    "S-实体类型_软件升级流程": 73,
    "S-实体类型_功能": 74,
    "S-实体类型_软件": 75,
    "S-实体类型_软件版本": 76,
    "S-实体类型_软件升级服务": 77,
    "S-非具体异常现象": 78,
    START_TAG: 79,
    STOP_TAG: 80
}

id2label = {_id: _label for _label, _id in list(tag_to_ix.items())}
# 标签一共有5个，所以embedding_dim为5
EMBEDDING_DIM = len(tag_to_ix)
# BILSTM隐藏层的特征数量，因为双向所以是2倍
HIDDEN_DIM = 4
model_dir = os.getcwd() + '/model.pth'


def preprocess(input_dir):
    """
    params:
        words：将json文件每一行中的文本分离出来，存储为words列表
        labels：标记文本对应的标签，存储为labels
    examples:
        words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
        labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
    """
    output_dir = './processedData.npz'
    # if os.path.exists(output_dir) is True:
    #     return
    word_list = []
    label_list = []
    data = []

    with open(input_dir, 'r', encoding='utf-8') as f:
        # 先读取到内存中，然后逐行处理
        for line in f.readlines():
            # loads()：用于处理内存中的json对象，strip去除可能存在的空格
            json_line = json.loads(line.strip())

            text = json_line['text']
            words = list(text)
            # 如果没有label，则返回None
            label_entities = json_line.get('entities', None)
            labels = ['O'] * len(words)

            if label_entities is not None:
                for label_index in label_entities:
                    start_index = label_index[0]
                    end_index = label_index[1]
                    label = label_index[2]
                    if start_index + 1 == end_index:
                        labels[start_index] = 'S-' + label
                    else:
                        labels[start_index] = 'B-' + label
                        labels[start_index + 1:end_index] = ['I-' + label] * (end_index - start_index - 1)
            word_list.append(words)
            label_list.append(labels)
            data.append((words, labels))

        with open("data.txt", "w") as f:  # 设置⽂件对象
            for i in data:  # 对于双层列表中的数据
                f.writelines(str(i) + '\n')

        # 保存成二进制文件
        np.savez_compressed(output_dir, words=word_list, labels=label_list)

        return data

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

###
# 根据labels对text提取结果
# 返回提取后的结果
##
def extract_result(text, labelsId, best_tag_score):
    """extract_result"""
    predict_tag = []
    for i in labelsId:
        predict_tag.append(id2label[i])

    start_idx = 0
    end_idx = 0
    entities = []
    is_start = 0
    i = 0
    cur_tag = ''
    temp_list = []
    for i, label in enumerate(predict_tag):
        if label[2:] != cur_tag:
            if cur_tag != '':
                start_idx = min(temp_list)
                end_idx = min(temp_list) + 1
                entities.append((''.join(text[start_idx:end_idx]), cur_tag, start_idx, end_idx, sigmoid_function(mean(best_tag_score[start_idx:end_idx])/1000)))
            cur_tag = label[2:]
            temp_list = []
            temp_list.append(i)
        else:
            temp_list.append(i)
    if cur_tag != '' and temp_list:
        start_idx = min(temp_list)
        end_idx = min(temp_list) + 1
        entities.append((''.join(text[start_idx:end_idx]), cur_tag, start_idx, end_idx, sigmoid_function(mean(best_tag_score[start_idx:end_idx])/1000)))

    for i, label in enumerate(predict_tag):

        if label.startswith(u"B-"):
            is_start += 1
            if is_start == 2:
                entities.append((text[start_idx:end_idx], label[2:], start_idx, end_idx, mean(best_tag_score[start_idx:end_idx])))
                is_start = 0
            start_idx = i
        if label.startswith(u"I-"):
            end_idx = i
        if label.startswith(u"S-"):
            entities.append((text[start_idx:end_idx], label[2:], start_idx, end_idx, mean(best_tag_score[start_idx:end_idx])))
    return entities


if __name__ == '__main__':
    # training_data_path = "./data/clue/train.jsonl"
    training_data_path = os.path.join(RESOURCES_DIR, 'ner','data','clue','train.jsonl')
    # Make up some training data
    training_data = preprocess(training_data_path)

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    # 首先是用未训练过的模型随便预测一个结果
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(3):
        for sentence, tags in training_data:
            # 训练前将梯度清零
            optimizer.zero_grad()

            # 准备输入
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # 前向传播，计算损失函数
            loss = model.neg_log_likelihood(sentence_in, targets)

            # 反向传播计算loss的梯度
            loss.backward()
            # 通过梯度来更新模型参数
            optimizer.step()

    torch.save(
        {
            # "epoch": epoch,
            "model": model.state_dict(),
            # "best_score": best_score,
            # "epochs_count": epochs_count,
            # "train_losses": train_losses,
            # "valid_losses": valid_losses,
            # "label_set": label_name,
            "word_to_ix": word_to_ix,
            "optimizer": optimizer
        },
        model_dir
        )

    # 使用训练过的模型来预测一个序列，与之前形成对比
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        path_score, tag_seq, best_tag_score = model(precheck_sent)
        print(extract_result(training_data[0][0], tag_seq, best_tag_score))
