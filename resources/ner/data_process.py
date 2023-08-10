import os
import json
import logging
import numpy as np


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            words：将json文件每一行中的文本分离出来，存储为words列表
            labels：标记文本对应的标签，存储为labels
        examples:
            words示例：['生', '生', '不', '息', 'C', 'S', 'O', 'L']
            labels示例：['O', 'O', 'O', 'O', 'B-game', 'I-game', 'I-game', 'I-game']
        """
        input_dir = self.data_dir + str(mode) + '.jsonl'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        with open(input_dir, 'r', encoding='utf-8') as f:
            # 先读取到内存中，然后逐行处理
            c=0
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
                    c += 1
                    print(c)
                word_list.append(words)
                label_list.append(labels)
                # 保存成二进制文件
            np.savez_compressed(output_dir, words=word_list, labels=label_list)
            logging.info("--------{} data process DONE!--------".format(mode))


if __name__ == '__main__':
    import config
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")