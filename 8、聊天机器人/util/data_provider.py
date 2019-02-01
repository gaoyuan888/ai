import os

import codecs
import collections
from tqdm import tqdm
import numpy as np


class AuxCode(object):
    """
    PAD,GO,EOS字符
    """
    def __init__(self):
        self.pad = 0
        self.go = 1
        self.eos = 2

    def get_pad_go_es(self):
        '''
        返回pad, go, es的值
        '''
        return self.pad, self.go, self.eos


class BatchData(object):
    """
    单个batch训练的数据结构
    """
    def __init__(self, query_input, query_length, answer_input, answer_length, answer_target):
        self.query_input = query_input
        self.query_length = query_length
        self.answer_input = answer_input
        self.answer_length = answer_length
        self.answer_target = answer_target

    def get_batch_size(self):
        '''
        取得batch的长度
        '''
        return len(self.query_input)


class DataProvider(object):
    """
    生成模型的数据准备类
    """
    def __init__(self, chat_field_num=2):
        self.chat_field_num = chat_field_num
        self.querys = []
        self.answers = []
        # 词汇表
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        self.pad, self.go, self.eos = AuxCode().get_pad_go_es()

    def load_chat_data(self, fileName):
        '''
        加载闲聊数据
        '''
        with codecs.open(fileName, "r", "utf-8") as f_in:
            for line in tqdm(f_in, desc="load chat data from " + fileName):
                line_array = [section.strip() for section in line.strip().split("\t")]
                if len(line_array) != self.chat_field_num:
                    print("data_format error: %s" % line.encode("utf-8"))
                    continue
                query = line_array[0]
                answer = line_array[1]

                self.querys.append(query)
                self.answers.append(answer)

    def make_corpus(self):
        '''
        形成语料库
        '''
        all_chars = []
        for query in tqdm(self.querys, desc="make query corpus"):
            all_chars += [char for char in query]
        for answer in tqdm(self.answers, desc="make answer corpus"):
            all_chars += [char for char in answer]

        # 计数
        counter = collections.Counter(all_chars)
        sorted_counter = sorted(counter.items(), key=lambda x: -x[1])
        all_chars, _ = zip(*sorted_counter)

        # 向量化, 默认PAD为0, GO为1，EOS为2
        self.char_to_id = dict(zip(all_chars, range(3, len(all_chars) + 3)))
        self.id_to_char = dict(zip(range(3, len(all_chars) + 3), all_chars))
        self.vocab_size = len(self.char_to_id) + 3

    def prepare_to_train(self):
        '''
        形成语料编码
        '''
        new_querys = []
        for query in tqdm(self.querys, desc="vectorize_query_data"):
            query = [self.char_to_id[char] for char in query]
            new_querys.append(query)
        self.querys = new_querys

        new_answers = []
        for answer in tqdm(self.answers, desc="vectorize_answer_data"):
            answer = [self.char_to_id[char] for char in answer]
            new_answers.append(answer)
        self.answers = new_answers

        # shuffle data
        dice = np.array([_ for _ in range(len(self.querys))], dtype=np.int32)
        np.random.shuffle(dice)

        new_querys = []
        new_answers = []

        for index in dice:
            new_querys.append(self.querys[index])
            new_answers.append(self.answers[index])

        self.querys = new_querys
        self.answers = new_answers

    def vectorize_for_inference(self, batch_query, vocab_id_map=None):
        '''
        为推断过程生成数字形式的样本
        '''
        if vocab_id_map is None:
            vocab_id_map = self.char_to_id

        batch_query = [query.strip() for query in batch_query]
        batch_query_ids = []
        for query in batch_query:
            batch_query_ids.append([vocab_id_map[c] for c in query
                if c in vocab_id_map and vocab_id_map[c] > self.eos])
        batch_length = [len(sentence_ids) for sentence_ids in batch_query_ids]
        max_length = max(batch_length)
        batch_query_ids = [(query_ids + [self.pad] * (max_length - len(query_ids)))
            for query_ids in batch_query_ids]
        return batch_query_ids, batch_length

    def gen_next_batch(self, batch_size):
        '''
        批量生成
        '''
        for i in range(len(self.querys) // batch_size):
            start = i*batch_size
            end = (i + 1)*batch_size
            query_max_len = max([len(_) for _ in self.querys[start:end]])
            answer_max_len = max([len(_) for _ in self.answers[start:end]]) + 1

            query_input = [query for query in self.querys[start:end]]
            query_length = [len(query) for query in query_input]
            query_input = [(query + [self.pad] * (query_max_len - len(query)))
                for query in query_input]

            answer_input = [([self.go] + answer) for answer in self.answers[start:end]]
            answer_length = [len(answer) for answer in answer_input]
            answer_input = [(answer + [self.pad] * (answer_max_len - len(answer)))
                for answer in answer_input]
            answer_target = [(answer + [self.eos]) for answer in self.answers[start:end]]
            answer_target = [(answer + [self.pad] * (answer_max_len - len(answer)))
                for answer in answer_target]

            batch_data = BatchData(np.array(query_input, dtype=np.int32),
                np.array(query_length, dtype=np.int32), np.array(answer_input, dtype=np.int32),
                np.array(answer_length, dtype=np.int32), np.array(answer_target, dtype=np.int32)
            )
            yield batch_data


def main():
    chat_data_path = os.path.join("..", "data", "chat_qa.txt")
    manager = DataProvider(2)
    manager.load_chat_data(chat_data_path)
    manager.make_corpus()
    manager.prepare_to_train()


if __name__ == "__main__":
    main()
