from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.tokenizers import Token, Tokenizer
from allennlp.data.fields import Field, TextField, MetadataField, MultiLabelField, ListField, LabelField
from typing import List, Dict
import tempfile
import torch
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.fields import TextField, MultiLabelField, ListField, Field, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask, masked_softmax
from allennlp.training.metrics import F1Measure, Average, Metric, FBetaMeasure

import json
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.data.tokenizers import Tokenizer
import torch.nn.functional as F
from DataRead import *
import pickle
import copy
sym_size = 160
@DatasetReader.register("mds_reader")
class TextClassificationTxtReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 model: str = None) -> None:#:是注释，表明参数是什么类型

        super().__init__(lazy=False)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._model = model

    @overrides
    def _read(self, file_path: str):
        with open('../data/160_last_topic2num.pk', 'rb') as f:
            topic2num = pickle.load(f)
        #读取训练数据
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)#dataset是个列表，里面的元素是列表
            for dialog in dataset:#取dataset里的元素为列表dialog，就是一个历史对话，包含多个字典，参见train.txt
                new_dialog = []
                history = []
                now_topic = []
                his_topic = []
                for sen in dialog:#取历史对话里面的每一个字典，就是一条话语的数据信息
                    # aa = sen['sym'] + sen['prop'] + sen['check'] + sen['diease'] + sen['medical']#把五个实体里的内容，就是标签，合在一起为一个列表
                    aa = sen['Symptom'] + sen['Attribute'] + sen['Test'] + sen['Disease'] + sen['Medicine']  # 把五个实体里的内容，就是标签，合在一起为一个列表
                    if len(aa) > 0:#如果标签不为空
                        if len(history) > 0 and sen['id'] == 'Doctor':#sen['id'] == 'doc':#如果history不为空且是医生说的话
                            #deepcopy独立复制出来
                            #history表示医生说的话之前的对话，next_sym表示现在医生要说的话的标签，now_topic表示医生说的话之前所有的标签
                            #new_dialog是列表里面是字典
                            new_dialog.append({"history": copy.deepcopy(history), "next_sym": copy.deepcopy(aa),
                                               'now_topic': copy.deepcopy(now_topic)})
                        now_topic.extend(aa)
                        his_topic.extend(aa)#his_topic包含所有的标签
                    #标签为空的情况，直接把对话加入history里
                    # history.append(sen['sentence'])
                    history.append(sen['Sentence'])
                for dic in new_dialog:#dic是new_dialog列表里的字典
                    future = copy.deepcopy(his_topic[len(dic['now_topic']):])#future为一个列表，将医生回复时的标签除外，医生回复之后的标签给future
                    dic['future'] = [topic2num[i] for i in future]
                    dic['next_sym'] = [topic2num[i] for i in dic['next_sym']]#应该是标签的序号
                    yield self.text_to_instance(dic)

    @overrides
    def text_to_instance(self, sample) -> Instance:#箭头是注释表明返回值是什么类型
        fields: Dict[str, Field] = {}
        tailored_history = sample['history']
        context = '。'.join(tailored_history)

        text_tokens = self._tokenizer.tokenize(context[-510:])
        fields['text'] = TextField(text_tokens, self._token_indexers)

        fileds_list = []
        for sen in tailored_history:
            sen = ' '.join(sen)
            txt_token = self._tokenizer.tokenize(sen)
            ff = TextField(txt_token, self._token_indexers)
            fileds_list.append(ff)
        fields["label"] = MultiLabelField(list(sample['next_sym']), skip_indexing=True, num_labels=sym_size)
        fields["future"] = MultiLabelField(list(sample['future']), skip_indexing=True, num_labels=sym_size)

        return Instance(fields)


@Model.register('symptoms_predictor')
class MyModel(Model):
    def __init__(self,
                text_field_embedder: TextFieldEmbedder,
                vocab: Vocabulary,
                seq2vec_encoder : Seq2VecEncoder = None,
                dropout: float = None,
                regularizer: RegularizerApplicator = None,
                ):
        super().__init__(vocab, regularizer)

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self.sym_size = sym_size
        self.embeddings = text_field_embedder
        self.vec_encoder = seq2vec_encoder
        self.hidden_dim = self.vec_encoder.get_output_dim()
        self.linear_class = torch.nn.Linear(self.hidden_dim, self.sym_size)
        # self.f_linear = torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
        self.dim = [12, 62, 4, 40, 62]
        self.true_list = [Average() for i in range(5)]
        self.pre_total = [Average() for i in range(5)]
        self.pre_true = [Average() for i in range(5)]
        self.total_pre = Average()
        self.total_true = Average()
        self.total_pre_true = Average()
        self.total_future_true = Average()
        self.macro_f = MacroF(self.sym_size)
        self.turn_acc = Average()

        self.future_acc = Average()

    def forward(self, text, label, **args):
        bs = label.size(0)
        embeddings = self.embeddings(text)  # bs  * seq_len * embedding

        mask = get_text_field_mask(text)  # bs * sen_num * sen_len

        seq_hidden = self.vec_encoder(embeddings, mask)  # bs , embedding
        # Shape: (batch_size, num_labels)
        topic_probs = F.sigmoid(self.linear_class(seq_hidden))
        # topic_weight = torch.ones_like(label) + 2 * label
        topic_weight = torch.ones_like(label) + label * 4
        loss = F.binary_cross_entropy(topic_probs, label.float(), topic_weight.float())
        # loss = F.binary_cross_entropy(topic_probs, label.long(), topic_weight.long())
        output_dict = {'loss': loss, 'probs': topic_probs, 'last_hidden': seq_hidden}
        # _, max_index = torch.max(topic_probs, -1)
        total_pre_list = []
        total_true_list = []
        total_pre_true_list = []
        pre_index = (topic_probs > 0.5).long()
        # pre_index = (topic_probs > 0.5).float()
        total_pre = torch.sum(pre_index)
        total_true = torch.sum(label)
        mask_index = (label == 1).long()
        # mask_index = (label == 1).float()
        self.macro_f(pre_index.cpu(), label.cpu())
        true_positive = (pre_index == label).long() * mask_index
        # true_positive = (pre_index == label).float() * mask_index
        st = 0
        for i in range(5):
            total_pre_list.append(torch.sum(pre_index[:, st:st + self.dim[i]]))
            total_true_list.append(torch.sum(label[:, st:st + self.dim[i]]))
            total_pre_true_list.append(torch.sum(true_positive[:, st:st + self.dim[i]]))
            st += self.dim[i]

        turn_true_num = (torch.sum(true_positive, 1) == torch.sum(mask_index, 1)).long()
        # turn_true_num = (torch.sum(true_positive, 1) == torch.sum(mask_index, 1)).float()
        self.turn_acc(torch.sum(turn_true_num).item() / bs)
        pre_true = torch.sum(true_positive)

        self.total_pre(total_pre.float().item())
        self.total_true(total_true.float().item())
        self.total_pre_true(pre_true.float().item())
        self.total_future_true(torch.sum((pre_index == args['future']).long() * (args['future'] == 1).long()).item())
        # self.total_future_true(torch.sum((pre_index == args['future']) * (args['future'] == 1).float()).item())
        for i in range(5):
            self.pre_total[i](total_pre_list[i].float().item())
            self.pre_true[i](total_pre_true_list[i].float().item())
            self.true_list[i](total_true_list[i].float().item())

        return output_dict

    def get_metrics(self, reset=False):
        metrics = {}
        total_pre = self.total_pre.get_metric(reset=reset)
        total_pre_true = self.total_pre_true.get_metric(reset=reset)
        total_true = self.total_true.get_metric(reset=reset)
        total_futuer_true = self.total_future_true.get_metric(reset=reset)
        for i in range(5):
            pre_i = self.pre_total[i].get_metric(reset=reset)
            pre_true_i = self.pre_true[i].get_metric(reset=reset)
            true_i = self.true_list[i].get_metric(reset=reset)
            acc_i, rec_i, f_i = 0., 0., 0.
            if pre_i > 0:
                acc_i = pre_true_i / pre_i
            if true_i > 0:
                rec_i = pre_true_i / true_i
            if acc_i + rec_i > 0:
                f_i = 2 * acc_i * rec_i / (acc_i + rec_i)
            metrics['f1' + str(i)] = f_i
            metrics['rc' + str(i)] = rec_i
            metrics['ac' + str(i)] = acc_i
        acc, rec, f1, facc = 0., 0., 0., 0.
        if total_pre > 0:
            acc = total_pre_true / total_pre
            facc = total_futuer_true / total_pre
        if total_true > 0:
            rec = total_pre_true / total_true
        if acc + rec > 0:
            f1 = 2 * acc * rec / (acc + rec)

        metrics['acc'] = acc
        metrics['rec'] = rec
        metrics['f1'] = f1
        metrics['macro_f1'] = self.macro_f.get_metric(reset=reset)
        metrics['turn_acc'] = self.turn_acc.get_metric(reset=reset)
        metrics['future_acc'] = facc
        return metrics

