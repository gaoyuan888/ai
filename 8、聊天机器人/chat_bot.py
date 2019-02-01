import os
import pickle

import codecs
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest

from util import data_provider


# 训练级参数
HYPERPARAMETERS = {"MAX_SEQ_LEN": 32, "EMBEDDING_SIZE": 128, "ENCODER_HIDDEN_SIZE": 128,
    "DECODER_HIDDEN_SIZE": 128, "ATTENTION_SIZE": 128, "USE_BEAMSEARCH": 1, "BEAM_WIDTH": 3,
    "LR_RATE": 1e-2, "KEEP_PROB": 0.5, "EPOCHS": 10, "BATCH_SIZE": 10, "STEP_TO_SAVE": 10
}

MAX_SEQ_LEN = HYPERPARAMETERS["MAX_SEQ_LEN"]
EMBEDDING_SIZE = HYPERPARAMETERS["EMBEDDING_SIZE"]
ENCODER_HIDDEN_SIZE = HYPERPARAMETERS["ENCODER_HIDDEN_SIZE"]
DECODER_HIDDEN_SIZE = HYPERPARAMETERS["DECODER_HIDDEN_SIZE"]
ATTENTION_SIZE = HYPERPARAMETERS["ATTENTION_SIZE"]
USE_BEAMSEARCH = HYPERPARAMETERS["USE_BEAMSEARCH"]
BEAM_WIDTH = HYPERPARAMETERS["BEAM_WIDTH"]
LR_RATE = HYPERPARAMETERS["LR_RATE"]
KEEP_PROB = HYPERPARAMETERS["KEEP_PROB"]
EPOCHS = HYPERPARAMETERS["EPOCHS"]
BATCH_SIZE = HYPERPARAMETERS["BATCH_SIZE"]
STEP_TO_SAVE = HYPERPARAMETERS["STEP_TO_SAVE"]


class Seq2SeqModel(object):
    """
    序列到序列生成模型
    """
    def __init__(self, mode):
        self.mode = mode
        self.saver = None
        self.vocab_size = 0
        self.id_to_char = {}
        self.char_to_id = {}
        self.data_provider = data_provider.DataProvider()
        self.pad, self.go, self.eos = data_provider.AuxCode().get_pad_go_es()

    def load_train_data(self, chat_file):
        '''
        加载训练和测试数据
        '''
        self.data_provider.load_chat_data(chat_file)
        self.data_provider.make_corpus()
        self.data_provider.prepare_to_train()

        self.id_to_char = self.data_provider.id_to_char
        self.char_to_id = self.data_provider.char_to_id
        self.vocab_size = self.data_provider.vocab_size

    def dump_dictionary(self, pkl_file_path):
        '''
        dump到文件
        '''
        dictionary_info = {}
        dictionary_info["id_to_char"] = self.id_to_char
        dictionary_info["char_to_id"] = self.char_to_id
        dictionary_info["vocab_size"] = self.vocab_size
        with codecs.open(pkl_file_path, "wb") as f_out:
            pickle.dump(dictionary_info, f_out)

    def load_dictionary(self, pkl_file_path):
        '''
        dump到文件
        '''
        with codecs.open(pkl_file_path, "rb") as f_in:
            dictionary_info = pickle.load(f_in)
        self.id_to_char = dictionary_info["id_to_char"]
        self.char_to_id = dictionary_info["char_to_id"]
        self.vocab_size = dictionary_info["vocab_size"]

    def build_model(self):
        '''
        建立seq2seq模型
        '''
        self.query_input = tf.placeholder(tf.int32, [None, None])
        self.query_length = tf.placeholder(tf.int32, [None])

        self.answer_input = tf.placeholder(tf.int32, [None, None])
        self.answer_target = tf.placeholder(tf.int32, [None, None])
        self.answer_length = tf.placeholder(tf.int32, [None])
        self.batch_size = array_ops.shape(self.query_input)[0]

        if self.mode == "train":
            self.max_decode_step = tf.reduce_max(self.answer_length)
            self.sequence_mask = tf.sequence_mask(self.answer_length,
                self.max_decode_step, dtype=tf.float32)
        elif self.mode == "decode":
            self.max_decode_step = tf.reduce_max(self.query_length) * 10

        # input and output embedding
        self.embeddings_matrix = tf.Variable(tf.random_uniform([
            self.vocab_size, EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)

        self.query_embeddings = tf.nn.embedding_lookup(self.embeddings_matrix, self.query_input)
        self.answer_embeddings = tf.nn.embedding_lookup(self.embeddings_matrix, self.answer_input)

        # encoder process
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
            rnn.BasicLSTMCell(ENCODER_HIDDEN_SIZE), self.query_embeddings,
            sequence_length=self.query_length, dtype=tf.float32)

        # 通过beam search 加工出一批临时变量，后续复用
        batch_size, encoder_outputs, encoder_state, encoder_length = (self.batch_size,
            self.encoder_outputs, self.encoder_state, self.query_length)

        if self.mode == "decode":
            batch_size = batch_size * BEAM_WIDTH
            encoder_outputs = seq2seq.tile_batch(t=self.encoder_outputs, multiplier=BEAM_WIDTH)
            encoder_state = nest.map_structure(lambda s: seq2seq.tile_batch(
                t=s, multiplier=BEAM_WIDTH), self.encoder_state)
            encoder_length = seq2seq.tile_batch(t=self.query_length, multiplier=BEAM_WIDTH)

        # attention wrapper
        self.attention_mechanism = seq2seq.BahdanauAttention(num_units=ENCODER_HIDDEN_SIZE,
            memory=encoder_outputs, memory_sequence_length=encoder_length)
        self.decoder_cell = seq2seq.AttentionWrapper(rnn.BasicLSTMCell(DECODER_HIDDEN_SIZE),
            attention_mechanism=self.attention_mechanism, attention_layer_size=ATTENTION_SIZE)
        self.decoder_initial_state = self.decoder_cell.zero_state(batch_size=batch_size,
            dtype=tf.float32).clone(cell_state=encoder_state)

        self.decoder_dense = tf.layers.Dense(self.vocab_size, dtype=tf.float32, use_bias=False,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 如果是训练过程，使用training helper, 否则使用greedyhelper或beamsearch helper
        if self.mode == "train":
            training_helper = seq2seq.TrainingHelper(inputs=self.answer_embeddings,
                sequence_length=self.answer_length)
            training_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell, helper=training_helper,
                initial_state=self.decoder_initial_state, output_layer=self.decoder_dense)

            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                impute_finished=True, maximum_iterations=self.max_decode_step)
            self.decoder_logits = tf.identity(decoder_outputs.rnn_output)

            self.loss = seq2seq.sequence_loss(logits=decoder_outputs.rnn_output,
                targets=self.answer_target, weights=self.sequence_mask)
            self.sample_ids = decoder_outputs.sample_id

            self.optimizer = tf.train.AdamOptimizer(LR_RATE)
            self.train_op = self.optimizer.minimize(self.loss)

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
        elif self.mode == "decode":
            start_tokens = tf.ones([self.batch_size], tf.int32) * self.go
            end_token = self.eos

            # 在beam search的情况下，给beam search helper传递的值，不需要使用BEAM_WIDTH的tensor
            # 此处使用beam_search/greedy helper解码都可以，如果只回复1条时等价
            if USE_BEAMSEARCH:
                inference_decoder = seq2seq.BeamSearchDecoder(cell=self.decoder_cell,
                    embedding=self.embeddings_matrix, start_tokens=start_tokens,
                    end_token=end_token, initial_state=self.decoder_initial_state,
                    beam_width=BEAM_WIDTH, output_layer=self.decoder_dense)
                # 使用beam_search的时候，结果是predicted_ids, beam_search_decoder_output
                # predicted_ids: [batch_size, decoder_targets_length, beam_size]
                # beam_search_decoder_output: scores, predicted_ids, parent_ids
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=inference_decoder, maximum_iterations=self.max_decode_step)
                self.sample_ids = decoder_outputs.predicted_ids
                self.sample_ids = tf.transpose(self.sample_ids, perm=[0, 2, 1]) # 转置成行句子
            else:
                decoding_helper = seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                    end_token=end_token, embedding=self.embeddings_matrix)
                inference_decoder = seq2seq.BasicDecoder(cell=self.decoder_cell,
                    helper=decoding_helper, initial_state=self.decoder_initial_state,
                    output_layer=self.decoder_dense)
                # 不使用beam_search的时候，结果是rnn_outputs, sample_id,
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32
                self.decoder_outputs_decode, self.final_state, _ = seq2seq.dynamic_decode(
                    decoder=inference_decoder, maximum_iterations=self.max_decode_step)
                self.sample_ids = self.decoder_outputs_decode.sample_id

    def get_query_and_answer_by_id(self, query_ids, answer_ids):
        '''
        训练模型
        '''
        # process query
        query = []
        for cht in query_ids:
            if cht == self.pad:
                break
            elif cht == self.go or cht == self.eos:
                print("fatal_error: 'go' or 'eos' in query ids.")
                break
            else:
                query.append(self.id_to_char[cht])
        query = "".join(query)
        # process answer
        answer = []
        for pos, cht in enumerate(answer_ids):
            if pos == 0 and cht == self.go:
                continue
            elif cht == self.go:
                print ("fatal_error: 'go' at wrong position in answer ids.")
                break
            elif cht == self.pad or cht == self.eos:
                break
            else:
                answer.append(self.id_to_char[cht])
        answer = "".join(answer)
        return query, answer

    def save_parameters(self, session, model_dir, model_name):
        '''
        测试模型
        '''
        if not self.saver:
            self.saver = tf.train.Saver(max_to_keep=1)
        self.saver.save(session, os.path.join(model_dir, model_name))

    def load_parameters(self, session, model_dir):
        '''
        加载模型
        '''
        if not self.saver:
            self.saver = tf.train.Saver()
        check_point = tf.train.get_checkpoint_state(os.path.join(model_dir))
        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            self.saver.restore(session, check_point.model_checkpoint_path)
            print('Reloading model succeed.')
            return True
        print('Reloading model failed.')
        return False

    def train(self, model_dir, model_name, epochs, batch_size, step_to_save, is_debug):
        '''
        训练模型
        '''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                step = 0
                for batch_data in self.data_provider.gen_next_batch(batch_size):
                    feed_dict = {
                        self.query_input: batch_data.query_input,
                        self.query_length: batch_data.query_length,
                        self.answer_target: batch_data.answer_target,
                        self.answer_input: batch_data.answer_input,
                        self.answer_length: batch_data.answer_length
                    }

                    _, loss, _, sample_ids = sess.run([self.train_op, self.loss,
                        self.summary_op, self.sample_ids], feed_dict=feed_dict)
                    print("epoch: %s, step: %s, loss: %s" % (i, step, loss))

                    if is_debug and step % step_to_save == 0:
                        for j in range(min(3, len(sample_ids))):
                            query_id, answer_id = list(zip(batch_data.query_input, sample_ids))[j]
                            query, answer = self.get_query_and_answer_by_id(query_id, answer_id)
                            query, answer = query, answer
                            print("query: %s, answer: %s" % (query, answer))

                    if step % step_to_save == 0:
                        self.save_parameters(sess, model_dir, model_name)
                    step += 1

    def inference(self, model_dir, batch_query):
        '''
        测试模型
        '''
        with tf.Session() as sess:
            self.load_parameters(sess, model_dir)
            process_function = self.data_provider.vectorize_for_inference
            batch_query, batch_length = process_function(batch_query, self.char_to_id)

            feed_dict = {
                self.query_input: batch_query,
                self.query_length: batch_length,
            }
            ids = sess.run(self.sample_ids, feed_dict=feed_dict)

            if USE_BEAMSEARCH:
                for answers in ids:
                    answer = [self.id_to_char[cht] for cht in answers[0] if cht > self.eos]
                    answer = "".join(answer)
                    yield answer
            else:
                for answer in ids:
                    answer = [self.id_to_char[cht] for cht in answer if cht > self.eos]
                    answer = "".join(answer)
                    yield answer


def main():
    stage = "decode"  # train/decode

    model_dir = os.path.join(".", "model")
    model_name = "generate_model.ckpt"
    dictionary_file = os.path.join(".", "model", "dictionary.pkl")

    if stage == "train":
        chat_file = os.path.join(".", "data", "chat_qa.txt")
        seq2seq_model = Seq2SeqModel("train")
        seq2seq_model.load_train_data(chat_file)
        seq2seq_model.dump_dictionary(dictionary_file)
        seq2seq_model.build_model()
        seq2seq_model.train(model_dir, model_name, EPOCHS, BATCH_SIZE, STEP_TO_SAVE, True)
    elif stage == "decode":
        seq2seq_model = Seq2SeqModel("decode")
        seq2seq_model.load_dictionary(dictionary_file)
        seq2seq_model.build_model()
        for answer in seq2seq_model.inference(model_dir, ["你好&&&"]):
            print(answer)


if __name__ == "__main__":
    main()
