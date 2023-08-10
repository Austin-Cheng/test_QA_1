# -*- coding: utf-8 -*- 
# @Time : 2022/5/24 14:52 
# @Author : liqianlan
import json
import os
from enum import Enum

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'app', 'data')
RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')

# Entity extract config dir,containing dict,rule,model files
ENTITY_CONFIG_DIR = os.path.join(DATA_DIR, 'entity')
ENTITY_DICT_FILE = os.path.join(ENTITY_CONFIG_DIR, 'dict.txt')  # 实体识别词典
ENTITY_RULE_FILE = os.path.join(ENTITY_CONFIG_DIR, 'rule.txt')  # 实体识别规则文件
ENTITY_MODEL_FILE = os.path.join(ENTITY_CONFIG_DIR, 'model.pth')
TRANSLATION_CONFIG_DIR = os.path.join(DATA_DIR, 'trans')

# Intent recognition config dir,containing dict,rule,model files
INTENT_CONFIG_DIR = os.path.join(DATA_DIR, 'intent')
INTENT_RULE_FILE = os.path.join(INTENT_CONFIG_DIR, 'rule.json')
INTENT_MODEL_PATH = os.path.join(INTENT_CONFIG_DIR, 'intent_model')

# Entity link config dir,containing dict,rule,model files
ENTITY_LINK_DIR = os.path.join(DATA_DIR, 'link')
ENTITY_SYNONYM_FILE = os.path.join(ENTITY_LINK_DIR, 'synonym.txt')
ENTITY_VECTOR_DIR = os.path.join(ENTITY_LINK_DIR, 'entity_vectors')
ENTITY_VECTOR_DIR_LOCAL = os.path.join(ENTITY_LINK_DIR, 'entity_vectors_local')
ENTITY_LINK_MODEL_DIR = os.path.join(ENTITY_LINK_DIR, 'bert_model')

# bert模型目录 10.4.15.191:/mnt/sdb2/qianlan/chinese_wwm_ext_L-12_H-768_A-12
# 启动server
# conda activate bert_env
# bert-serving-start -model_dir=./chinese_wwm_ext_L-12_H-768_A-12/ -num_worker=1
BERT_SERVER_IP = os.getenv('BERT_SERVER_IP')  # 用于bert词向量相似度计算

# 实体识别模型阈值
NER_MODEL_THR = 0.99
# 实体链接词向量相似度阈值
LINK_SIM_THR_LCS = 0.5
LINK_SIM_THR = 0.9
# 意图识别模型阈值
INTENT_MODEL_THR = 0.99

GENERAL_INTENT_NAME = '查文档'
GENERAL_INTENT_SPACE = '文档'
UNRECOGNIZED_INTENT_NAME = '其他'
UNRECOGNIZED_INTENT_SPACE = '无'


class SearchGraphSpace(Enum):
    ks8asnodelete = '运维'
    concept = '概念'
    AB8 = '文档'
    none = '无'


# 标注实体类型为中文，实际预测实体类型为英文
ENT_TYPE_EN_CN_MAP = {
    "abnormal": "异常情况",
    "cate": "异常类别",
    "because": "异常原因",
    "software": "软件",
    "software_version": "软件版本",
    "function": "功能",
    "command": "命令",
    "command_describe": "命令描述",
    "option": "命令参数",
    "option_describe": "命令参数描述",
    "resolve": "解决方案",
    "upstep": "软件升级流程",

    "command_type": "实体类型_命令",
    "option_type": "实体类型_命令参数",
    "command_describe_type": "实体类型_命令描述",
    "option_describe_type": "实体类型_命令参数描述",
    "because_type": "实体类型_异常原因",
    "abnormal_type": "实体类型_异常现象",
    "cate_type": "实体类型_类别",
    "resolve_type": "实体类型_解决方案",
    "upstep_type": "实体类型_软件升级流程",
    "function_type": "实体类型_功能",
    "software_type": "实体类型_软件",
    "software_version_type": "实体类型_软件版本",
    "supgrade_type": "实体类型_软件升级服务",
    "unspecific_abnormal": "非具体异常现象",

}
ENT_TYPE_CN_EN_MAP = {v: k for k, v in ENT_TYPE_EN_CN_MAP.items()}

INTENT_PRED_ANNOTATION_MAP = {

    "软件升级异常分类原因解决方案查询__运维": "软件升级异常分类原因解决方案查询（两个软件版本）",
    "软件版本异常-分析类型、原因、处理方法（只有一个软件版本）__运维": "软件版本异常-分析类型、原因、处理方法（只有一个软件版本）",

    "升级异常-分析类型、原因、处理方法(只有软件，无软件版本）__运维": "升级异常-分析类型、原因、处理方法(只有软件，无软件版本）",

    "升级异常-分析类型、原因、处理方法(无异常情况具体信息）__运维": "升级异常-分析类型、原因、处理方法(无异常情况具体信息）",
    "异常情况-分析类型、原因、处理方法（无软件、有异常情况名称）__运维": "异常情况-分析类型、原因、处理方法（无软件、有异常情况名称）",
    "异常信息message-分析类型、原因、处理方法__运维": "异常信息message-分析类型、原因、处理方法",
    "升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）__运维": "升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）",
    "异常情况-查文档__文档": "异常情况-查文档",

    "软件升级流程查询__运维": "软件升级流程查询",
    "异常列表__运维": "异常列表（无具体异常信息）",

    "升级列表__运维": "升级列表",
    "通过作用查询命令__运维": "通过作用查询命令",
    "命令参数解释__运维": "命令参数解释",
    "命令描述__运维": "命令描述",
    "软件版本推理-最新__运维": "软件版本推理-最新",
    "软件版本推理-统计__运维": "软件版本推理-统计",
    "软件版本介绍__运维": "软件版本介绍",
    "名词解释__概念": "名词解释",
    "其他__无": "其他",
    "查文档__文档": "查文档",

}
INTENT_ANNOTATION_PRED_MAP = {v: k for k, v in INTENT_PRED_ANNOTATION_MAP.items()}

# 标注数据包含意图
annotated_intents = ['升级异常-分析类型、原因、处理方法(只有软件，无软件版本）',
                     '升级异常-分析类型、原因、处理方法(无软件名称、无异常情况名称）',
                     '异常列表（无具体异常信息）',
                     '软件升级异常分类原因解决方案查询（两个软件版本）', '软件升级流程查询']

# 翻译数据映射字典
transDict = {}
ENTITY_TRANSLATION_FILE = os.path.join(TRANSLATION_CONFIG_DIR, 'propm.json')
with open(ENTITY_TRANSLATION_FILE, 'r', encoding='utf-8') as f:
    transDict = json.loads(f.read())

# 显示格式类型全局静态变量
DISPLAY_TXT_TYPE = 'txt'
DISPLAY_DOC_TYPE = 'doc'

# 默认的分页参数
DEFAULT_LIMIT = 10
DEFAULT_OFFSET = 0
MAX_OFFSET = 100
