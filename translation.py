import unicodedata
import re
import random
import jieba
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from tqdm import tqdm
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
import os
import argparse
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# 每个pair中，制作出中文和英文词典
BOS_token = 0   # Begin Of Sentence
EOS_token = 1   # End Of Sentence

def get_parser() -> argparse.ArgumentParser:
    """Return an argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_length",
        default=20,
        type=int,
        help="the max length of instruction (default: 20)",
    )
    parser.add_argument(
        "--all_data",
        default=False,
        action="store_true",
        help="whether not to use all data (default: False)",
    )
    parser.add_argument(
        "--num_epochs",
        default=75,
        type=int,
        help="the num of train epochs (default: 75)",
    )
    parser.add_argument(
        "--save_epochs",
        default=25,
        type=int,
        help="the num of train epochs to save (default: 25)",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="the size of hidden (default: 256)",
    )
    parser.add_argument(
        "--num_workers",
        default=3,
        type=int,
        help="the num of workers (default: 3)",
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="the random (default: 1)",
    )
    parser.add_argument(
        "--reverse",
        default=False,
        action="store_true",
        help="whether not to reverse (default: False)",
    )
    parser.add_argument(
        "--logdir",
        default="results",
        type=str,
        help="the path to save data (default: 'results')",
    )
    parser.add_argument(
        "--train_rate",
        default=0.7,
        type=float,
        help="the rate to train (default: 0.7)",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.01,
        type=float,
        help="the learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--teacher_forcing_ratio",
        default=0.5,
        type=float,
        help="the rate to use teacher forcing (default: 0.5)",
    )
    return parser


### 数据预处理的主要步骤包括：
# 1. 读取txt文件，并按行分割，再把每一行分割成一个pair(Eng, Chinese)
# 2. 过滤并处理文本信息
# 3. 从每个pair中，制作出中文词典和英文词典
# 4. 构建训练集


### 过滤并处理文本信息
# 为了便于数据处理，把Unicode字符串转换为ASCII编码（同时过滤掉'Mn'容易报错的数据）
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 对英文转换为小写，去空格及非字母符号等处理
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)   # 替换，相当于找到所有.?!，在前面加个空格
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

### 定义字典的存储数据结构
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "BOS", 1: "EOS"}  # Begining Of Sequence, End Of Sequence
        self.n_words = 2  # Count BOS and EOS单词总量 初始化包含 BOS and EOS
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    #处理英文句子
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    #处理中文句子
    def addSentence_cn(self, sentence):
        for word in list(jieba.cut(sentence)):
            self.addWord(word)


### 读数据，这里标签lang1，lang2作为参数，可提高模块通用性，可以进行多种语言的互译，只需修改数据文件及这两个参数即可
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    #读文件，然后分成行
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    #把行分成句子对，并进行规范化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    #判断是否需要转换语句的次序，如[英文，中文]转换为[中文，英文]次序
    if reverse:
        pairs=[list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang =Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs

def filterPair(p, reverse, max_length, eng_prefixes):
    if reverse:
        return (len(list(jieba.cut(p[0]))) < max_length) and \
            (len(p[1].split(' ')) < max_length) and \
            p[1].startswith(eng_prefixes)
    else:
        return (len(p[0].split(' ')) < max_length) and \
            (len(list(jieba.cut(p[1]))) < max_length) and \
            p[0].startswith(eng_prefixes)

def filterPairs(pairs, reverse, max_length, eng_prefixes):
    return [pair for pair in pairs if filterPair(pair, reverse, max_length, eng_prefixes)]

# 把以上数据预处理函数，放在一起，实现对数据的预处理
def prepareData(lang1, lang2, reverse, max_length, eng_prefixes):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    
    print('Read %s sentence pairs' % len(pairs))
    pairs = filterPairs(pairs, reverse, max_length, eng_prefixes)
    print('Trimmed to %s sentence pairs' % len(pairs))
    print('Counting words...')
    for pair in pairs:
        if reverse:
            input_lang.addSentence_cn(pair[0])
            output_lang.addSentence(pair[1])
        else:
            input_lang.addSentence(pair[0])
            output_lang.addSentence_cn(pair[1])
    print('Counted words:')
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

### 数据集
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def indexesFromSentence_cn(lang, sentence):
    return [lang.word2index[word] for word in list(jieba.cut(sentence))]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)   # 每个句子结尾要加EOS
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorFromSentence_cn(lang, sentence):
    indexes = indexesFromSentence_cn(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair, reverse):
    if reverse:
        input_tensor = tensorFromSentence_cn(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
    else:
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence_cn(output_lang, pair[1])
    return (input_tensor, target_tensor, pair[0], pair[1])


class PairsDataset(Dataset):
    def __init__(
        self,
        pairs,
        reverse,
    ):
        self.pairs = [tensorsFromPair(pair, reverse) for pair in pairs]
        self.reverse = reverse

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        return self.pairs[index]


### 训练时所用到的函数定义
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)  # 实现词与词向量的映射，通俗来讲就是将文字转换为一串数字，作为训练的一层
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)   # 利用一个全连接层，自动计算attention的权重
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, device):    
    encoder_hidden = encoder.initHidden()   # hidden state
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    loss = 0

    encoder.zero_grad()
    decoder.zero_grad()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # encoder
    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # decoder
    decoder_input = torch.tensor([[BOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  # 返回最大的值及其位置下标
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # backward
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_pairs, writer, epoch, teacher_forcing_ratio):
    device = next(encoder.parameters()).device  # get the device id
    encoder.train()  # train mode (dropout and batchnorm is used)
    decoder.train()  # train mode (dropout and batchnorm is used)
    for step, training_pair in enumerate(tqdm(train_pairs, desc=f'Train Epoch {epoch}')):
        input_tensor = training_pair[0][0].to(device)
        target_tensor = training_pair[1][0].to(device)

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, device)
        global_step = step + epoch * len(train_pairs)
        writer.add_scalar("loss", loss, global_step=global_step)
        writer.add_scalar('learning_rate/encoder', encoder_optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('learning_rate/decoder', decoder_optimizer.param_groups[0]['lr'], epoch)


### 可视化
def evaluate(encoder, decoder, input_tensor, device):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        # encoder
        encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],  encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        # decoder
        decoder_input = torch.tensor([[BOS_token]], device=device)  # BOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(decoder.max_length, decoder.max_length)
        for di in range(decoder.max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            if decoder_input.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[decoder_input.item()])

        return decoded_words, decoder_attentions[:di + 1]


def evaluateIters(encoder, decoder, test_pairs, reverse):
    device = next(encoder.parameters()).device  # get the device id
    encoder.eval()  # eval mode (dropout and batchnorm is not used)
    decoder.eval()  # eval mode (dropout and batchnorm is not used)
    total_bleu = 0

    for pair in test_pairs:
        input_tensor = pair[0][0].to(device)
        output_words, attentions = evaluate(
            encoder, decoder, input_tensor, device)
        output_words.pop()
        if reverse:
            groudtruth = pair[3][0].split(" ")
        else:
            groudtruth = [word for word in list(jieba.cut(pair[3][0]))]
        bleu_score = sentence_bleu([output_words], groudtruth)
        total_bleu += bleu_score
    return total_bleu


def evaluateRandomly(encoder, decoder, pairs, reverse, n=20):
    device = next(encoder.parameters()).device  # get the device id

    for _ in range(n):
        pair = random.choice(pairs)
        if reverse:
            input_tensor = tensorFromSentence_cn(input_lang, pair[0]).to(device)
        else:
            input_tensor = tensorFromSentence(input_lang, pair[0]).to(device)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, input_tensor, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def showAttention(input_sentence, output_words, attentions, logdir=""):
    # Set up figure with colorbar
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + list(jieba.cut(input_sentence)) + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # plt.show()
    print(output_words)
    plt.savefig(os.path.join(logdir, 'result-%s.jpg' % input_sentence))


def evaluateAndShowAttention(input_sentence, encoder, decoder, reverse, logdir=""):
    device = next(encoder.parameters()).device  # get the device id
    if reverse:
        input_tensor = tensorFromSentence_cn(input_lang, input_sentence).to(device)
    else:
        input_tensor = tensorFromSentence(input_lang, input_sentence).to(device)
    output_words, attentions = evaluate(encoder, decoder, input_tensor, device)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions, logdir)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"use {device}")

    parser = get_parser()
    args = parser.parse_args()

    ### 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 为便于训练，这里选择部分数据
    if not args.all_data:
        # 限制句子开头
        eng_prefixes = (
            "i am ", "i'm ", 
            "he is", "he's ", 
            "she is", "she's ", 
            "you are", "you're ", 
            "we are", "we're ", 
            "they are", "they're "
        )
    else:
        # 不限制句子开头
        eng_prefixes = (
            "a", "b", "c", "d", "e", "f",
            "g", "h", "i", "j", "k", "l", 
            "m", "n", "o", "p", "q", "r",
            "s", "t", "u", "v", "w", "x",
            "y", "z", '"'
        )

    
    ### 运行预处理函数
    input_lang, output_lang, pairs = prepareData('eng', 'cmn', args.reverse, args.max_length, eng_prefixes)

    ### 数据集
    random.shuffle(pairs)   # 打乱数据集

    train_dataset = PairsDataset(pairs[: int(len(pairs)*args.train_rate)], args.reverse)
    print(f"length of train pairs: {len(train_dataset)}")
    test_dataset = PairsDataset(pairs[int(len(pairs)*args.train_rate): ], args.reverse)
    print(f"length of test pairs: {len(test_dataset)}")
    
    train_data_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=1,   # 只考虑bs=1的情况
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_data_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logdir = f"{args.logdir}/{input_lang.name}_{output_lang.name}_{args.max_length}length_{len(train_data_loader)}train_{args.num_epochs}epochs_{args.hidden_size}size"
    print(logdir)

    # the path to save models
    save_prefix = f"{logdir}/data"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    writer = SummaryWriter(logdir=f"{logdir}/tb", flush_secs=30)  # SummaryWriter

    ### 开始训练
    print('training...')

    encoder = EncoderRNN(input_lang.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, output_lang.n_words, max_length=args.max_length, dropout_p=0.1).to(device)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()    # 定义损失函数，输入是一个对数概率向量和一个目标标签，CrossEntropyLoss()=log_softmax() + NLLLoss() 
    
    best_train_bleu = 0
    best_test_bleu = 0
    
    for epoch in range(args.num_epochs):
        trainIters(encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, train_data_loader, writer, epoch, args.teacher_forcing_ratio)

        ### 保存模型
        model_path = f"{save_prefix}/{epoch}.bin"
        if (epoch + 1) % args.save_epochs == 0 or (epoch + 1) == args.num_epochs:
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )

        ### 测试模型
        bleu = evaluateIters(encoder, decoder, train_data_loader, args.reverse)/len(train_data_loader)
        writer.add_scalar("train/bleu", bleu, global_step=epoch*len(train_data_loader))
        if bleu >= best_train_bleu:
            best_train_bleu = bleu
            model_path = f"{save_prefix}/best.bin"
            
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )

        bleu = evaluateIters(encoder, decoder, test_data_loader, args.reverse)/len(test_data_loader)
        writer.add_scalar("val/bleu", bleu, global_step=epoch*len(test_data_loader))
        if bleu >= best_test_bleu:
            best_test_bleu = bleu
            model_path = f"{save_prefix}/best.bin"
            
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )

        print(f"epoch:{epoch}\tbest_train_bleu: {best_train_bleu}\tbest_test_bleu: {best_test_bleu}")


    ### 分析结果
    evaluateRandomly(encoder, decoder, pairs, args.reverse, 20)
    testsentence = "我很高兴。" if args.reverse else "i am happy ."
    evaluateAndShowAttention(testsentence, encoder, decoder, args.reverse, logdir)
