# -*- coding: utf-8 -*- 
# @Time : 2022/5/27 14:58 
# @Author : liqianlan
import os
import pickle
import random

import numpy as np


def validate(mentions, mention):
    """
    Judge if mention has conflict with mentions in location.
    Args:
        mentions: Mentions already exist.
        mention: New mention extracted.

    Returns: True, or False.

    """
    for mt in mentions:
        if mention.start >= mt.end or mention.end <= mt.start:
            continue
        return False
    return True


def get_lcs_len(x, y):
    """
    最长公共子序列长度

    Args:
        x:
        y:

    Returns:

    """
    dp = [[0] * len(y) for _ in range(len(x))]
    for j in range(len(y)):
        if y[j] == x[0]:
            dp[0][j] = 1
        else:
            dp[0][j] = dp[0][j - 1] if j > 0 else 0

    for i in range(len(x)):
        if i == 0:
            if y[0] == x[i]:
                dp[i][0] = 1
        else:
            if y[0] == x[i]:
                dp[i][0] = 1
            else:
                dp[i][0] = dp[i - 1][0]
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            if x[i] == y[j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]


def LCS(x, y):
    c = np.zeros((len(x) + 1, len(y) + 1))
    b = np.zeros((len(x) + 1, len(y) + 1))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                c[i, j] = c[i - 1, j - 1] + 1
                b[i, j] = 2
            else:
                if c[i - 1, j] >= c[i, j - 1]:
                    c[i, j] = c[i - 1, j]
                    b[i, j] = 1
                else:
                    c[i, j] = c[i, j - 1]
                    b[i, j] = 3
    return c, b


def get_lcs(x, y):
    c, b = LCS(x, y)
    i = len(x)
    j = len(y)
    lcs = ''
    while i > 0 and j > 0:
        if b[i][j] == 2:
            lcs = x[i - 1] + lcs
            i -= 1
            j -= 1
        if b[i][j] == 1:
            i -= 1
        if b[i][j] == 3:
            j -= 1
        if b[i][j] == 0:
            break
    lcs_len = len(lcs)
    return lcs, lcs_len


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    # vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def load_kernel_bias(path):
    with open(os.path.join(path, 'kernel_bias.pkl'), 'rb') as f:
        data = pickle.load(f)

    kernel = data[0]
    bias = data[1]
    return kernel, bias


def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串
    """
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str
