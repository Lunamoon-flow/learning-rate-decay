import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.backends.cudnn as cudnn
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
import torch.nn as nn
import math
from peft import get_peft_model, AdaLoraConfig, TaskType
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset,Dataset,DataLoader
from torch.utils.data import RandomSampler,BatchSampler
from torch.optim.lr_scheduler import LambdaLR
# def init_seeds(seed=0, cuda_deterministic=True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
#     if cuda_deterministic:  # slower, more reproducible
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     else:  # faster, less reproducible
#         cudnn.deterministic = False
#         cudnn.benchmark = True




from torch.utils.data import Dataset, DataLoader
from copy import deepcopy





import pandas as pd
import numpy as np







# def find_all_linear_names(model):
#     """
#     找出所有全连接层，为所有全连接添加adapter
#     """
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     for name, module in model.named_modules():
#         if isinstance(module, cls):
#             names = name.split('.')
#             lora_module_names.add(names[0] if len(names) == 1 else names[-1])
#
#     if 'lm_head' in lora_module_names:  # needed for 16-bit
#         lora_module_names.remove('lm_head')
#     return list(lora_module_names)

from peft import prepare_model_for_kbit_training


from peft import AdaLoraConfig
import os
import numpy
#import tqdm


if __name__ == '__main__':






    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
    local_rank = args.local_rank
    # if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    dl_num=dist.get_rank()

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(local_rank)
    # rank = torch.distributed.get_rank()
    # init_seeds(3407 + rank)




    model_name_or_path = '/root/Baichuan2-7B-Chat'

    #device_model = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model_train = AutoModelForCausalLM.from_pretrained(model_name_or_path,quantization_config=bnb_config,
                                                       trust_remote_code=True)
    # model_train = nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
    #                                                  find_unused_parameters=True)
    model_train.generation_config = GenerationConfig.from_pretrained(model_name_or_path)


    def find_index(tensor, target):
        # 查找指定元素在张量中的索引
        indices = torch.where(tensor == target)[0]
        if indices.numel() > 0:
            # 返回最后一个索引
            return indices
        else:
            # 如果没有找到指定元素，返回-1
            return -1


    def last_index(tensor, target):
        # 查找指定元素在张量中的索引
        indices = torch.where(tensor == target)[0]
        if indices.numel() > 0:
            # 返回最后一个索引
            return indices[-1].item()
        else:
            # 如果没有找到指定元素，返回-1
            return -1


    def build_chat_input(messages, model=model_train,
                         tokenizer=tokenizer, pretrain=1):
        # max_new_tokens = None
        # max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
        # max_input_tokens = model.config.max_position_embeddings - max_new_tokens
        # max_input_tokens = max(model.config.max_position_embeddings // 2, max_input_tokens)
        max_input_tokens = model.config.model_max_length


        total_input, round_input, total_label, round_label = [], [], [], []

        for i, message in enumerate(messages[::-1]):
            content_tokens = tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = content_tokens + round_input
                if pretrain == 0:
                    round_label = [-100 for _ in content_tokens] + round_label
                else:
                    round_label = content_tokens + round_label

                if total_input and len(total_input) + len(round_input) > max_input_tokens:
                    break
                else:
                    total_input = round_input + total_input
                    total_label = round_label + total_label
                    if len(total_input) >= max_input_tokens:
                        break
                    else:
                        round_input = []
                        round_label = []

            elif message['role'] == 'assistant':
                # round_input = content_tokens + [
                #     model.generation_config.eos_token_id
                # ] + round_input
                round_input = content_tokens + round_input

                # round_label = content_tokens + [
                #     model.generation_config.eos_token_id  # 注意，除了要学习机器人回复内容，还要学习一个结束符。
                # ] + round_label
                round_label = content_tokens + round_label

            else:
                raise ValueError(f"message role not supported yet: {message['role']}")

        total_input = total_input[-max_input_tokens:]  # truncate left
        total_label = total_label[-max_input_tokens:]
        # print(total_input)
        # print(total_label)

        # total_input.append(model.generation_config.assistant_token_id)
        return total_input, total_label


    def data_collator(examples: list, data_strong=1, model=model_train, max_length=1500):
        len_batch = len(examples)
        len_ids = [len(example["input_ids"]) for example in examples]
        longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

        input_ids = []
        labels_list = []
        eos_id = model.generation_config.eos_token_id

        # for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        #     ids = example["input_ids"]
        #     labs = example["labels"]
        #     # ids = ids + [151643] * (longest - length)
        #
        #     #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
        #     #             labs = labs + [-100] * (longest - length)
        #     ids = ids
        #     labs = labs
        #
        #     input_ids.append(torch.LongTensor(ids))
        #
        #     labels_list.append(torch.LongTensor(labs))
        #
        # print(input_ids)

        if data_strong == 1:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for example in examples:
                ids = example["input_ids"]
                labs = example["labels"]
                # ids = ids + [151643] * (longest - length)

                #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
                #             labs = labs + [-100] * (longest - length)
                ids = ids
                labs = labs
                len_ids_2 = len(ids)
                len_labs = len(labs)
                input_ids.append(torch.LongTensor(ids).view(1, len_ids_2))
                labels_list.append(torch.LongTensor(labs).view(1, len_labs))

            input_ids = torch.cat(input_ids, dim=1)
            labels = torch.cat(labels_list, dim=1)
            numer_input = input_ids.numel()
            numer_label = labels.numel()
            # print("numer_input",numer_input)
            if input_ids.numel() >= max_length:
                # print(input_ids)
                # print(input_ids[0, :max_length])
                input_ids = input_ids[0, :max_length]
                labels = labels[0, :max_length]
                input_ids = input_ids.view(1, input_ids.numel())
                labels = labels.view(1, input_ids.numel())
                input_ids = input_ids.long()
                labels = labels.long()
            else:
                buquan = max_length - input_ids.numel()
                awb = torch.LongTensor([eos_id for _ in range(buquan)]).view(1, buquan)
                input_ids = torch.cat((input_ids, awb), dim=1)
                input_ids = input_ids.view(1, input_ids.numel())
                labels = torch.cat((labels, awb), dim=1)
                labels = labels.view(1, labels.numel())
                input_ids = input_ids.long()
                labels = labels.long()



        else:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
                ids = example["input_ids"]
                labs = example["labels"]
                ids = ids + [eos_id] + [eos_id] * (longest - length)

                # ids = ids + [model.generation_config.pad_token_id] * (longest - length)
                labs = labs + [eos_id] + [-100] * (longest - length)
                # ids = ids
                # labs = labs

                input_ids.append(torch.LongTensor(ids))
                labels_list.append(torch.LongTensor(labs))

            # input_ids.append(torch.LongTensor(eos_id))
            # labels_list.append(torch.LongTensor(eos_id))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels_list)
            # eos_id = model.generation_config.eos_token_id
            # T1 = torch.LongTensor([eos_id for _ in range(len_batch)]).view(len_batch, 1)
            # a = []
            # a.append(input_ids)
            # a.append(T1)
            # b = []
            # b.append(labels)
            # b.append(T1)
            #
            # # a.append(T2)
            # # a.append(T3)
            #
            # input_ids = torch.cat(a, dim=1)
            # labels = torch.cat(b, dim=1)

        # print(input_ids)
        # print(input_ids.numel())
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


    def data_collator_2(examples: list, data_strong=0, model=model_train, max_length=1024):
        len_batch = len(examples)
        len_ids = [len(example["input_ids"]) for example in examples]
        longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

        input_ids = []
        labels_list = []
        eos_id = model.generation_config.eos_token_id
        pad_token_id = model.generation_config.pad_token_id

        # for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        #     ids = example["input_ids"]
        #     labs = example["labels"]
        #     # ids = ids + [151643] * (longest - length)
        #
        #     #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
        #     #             labs = labs + [-100] * (longest - length)
        #     ids = ids
        #     labs = labs
        #
        #     input_ids.append(torch.LongTensor(ids))
        #
        #     labels_list.append(torch.LongTensor(labs))
        #
        # print(input_ids)

        if data_strong == 1:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
                ids = example["input_ids"]
                labs = example["labels"]
                # ids = ids + [151643] * (longest - length)

                #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
                #             labs = labs + [-100] * (longest - length)
                ids = ids
                labs = labs
                len_ids = len(ids)
                len_labs = len(labs)
                input_ids.append(torch.LongTensor(ids).view(1, len_ids))
                labels_list.append(torch.LongTensor(labs).view(1, len_labs))

            input_ids = torch.cat(input_ids, dim=1)
            labels = torch.cat(labels_list, dim=1)
            numer_input = input_ids.numel()
            numer_label = labels.numel()
            # print("numer_input",numer_input)
            if input_ids.numel() >= max_length:
                # print(input_ids)
                # print(input_ids[0, :max_length])
                input_ids = input_ids[0, :max_length]
                labels = labels[0, :max_length]
                input_ids = input_ids.view(1, input_ids.numel())
                labels = labels.view(1, input_ids.numel())
            else:
                buquan = max_length - input_ids.numel()
                awb = torch.LongTensor([eos_id for _ in range(buquan)]).view(1, buquan)
                input_ids = torch.cat((input_ids, awb), dim=1)
                input_ids = input_ids.view(1, input_ids.numel())
                labels = torch.cat((labels, awb), dim=1)
                labels = labels.view(1, labels.numel())


        else:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
                ids = example["input_ids"]
                labs = example["labels"]
                ids = ids + [pad_token_id] + [pad_token_id] * (longest - length)

                # ids = ids + [model.generation_config.pad_token_id] * (longest - length)
                labs = labs + [pad_token_id] + [-100] * (longest - length)
                # ids = ids
                # labs = labs

                input_ids.append(torch.LongTensor(ids))
                labels_list.append(torch.LongTensor(labs))

            # input_ids.append(torch.LongTensor(eos_id))
            # labels_list.append(torch.LongTensor(eos_id))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels_list)
            # eos_id = model.generation_config.eos_token_id
            # T1 = torch.LongTensor([eos_id for _ in range(len_batch)]).view(len_batch, 1)
            # a = []
            # a.append(input_ids)
            # a.append(T1)
            # b = []
            # b.append(labels)
            # b.append(T1)
            #
            # # a.append(T2)
            # # a.append(T3)
            #
            # input_ids = torch.cat(a, dim=1)
            # labels = torch.cat(b, dim=1)

        # print(input_ids)
        # print(input_ids.numel())
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def data_collator_train(examples: list, data_strong=1, model=model_train, max_length=1024):
        len_batch = len(examples)
        len_ids = [len(example["input_ids"]) for example in examples]
        longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

        input_ids = []
        labels_list = []
        eos_id = model.generation_config.eos_token_id
        pad_token_id = model.generation_config.pad_token_id

        # for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        #     ids = example["input_ids"]
        #     labs = example["labels"]
        #     # ids = ids + [151643] * (longest - length)
        #
        #     #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
        #     #             labs = labs + [-100] * (longest - length)
        #     ids = ids
        #     labs = labs
        #
        #     input_ids.append(torch.LongTensor(ids))
        #
        #     labels_list.append(torch.LongTensor(labs))
        #
        # print(input_ids)

        if data_strong == 1:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for example in examples:
                ids = example["input_ids"]
                labs = example["labels"]
                # ids = ids + [151643] * (longest - length)

                #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
                #             labs = labs + [-100] * (longest - length)
                ids = ids
                labs = labs
                len_ids = len(ids)
                len_labs = len(labs)
                input_ids.append(torch.LongTensor(ids).view(1, len_ids))
                labels_list.append(torch.LongTensor(labs).view(1, len_labs))


            input_ids = torch.cat(input_ids, dim=1)
            labels = torch.cat(labels_list, dim=1)
            len_eos = 1
            awb = torch.LongTensor([eos_id for _ in range(len_eos)]).view(1, len_eos)
            input_ids = torch.cat((input_ids, awb), dim=1)
            input_ids = input_ids.view(1, input_ids.numel())
            labels = torch.cat((labels, awb), dim=1)
            labels = labels.view(1, labels.numel())

            input_ids = input_ids.long()
            labels = labels.long()
            # numer_input = input_ids.numel()
            # numer_label = labels.numel()
            # # print("numer_input",numer_input)
            # if input_ids.numel() >= max_length:
            #     # print(input_ids)
            #     # print(input_ids[0, :max_length])
            #     input_ids = input_ids[0, :max_length]
            #     labels = labels[0, :max_length]
            #     input_ids = input_ids.view(1, input_ids.numel())
            #     labels = labels.view(1, input_ids.numel())
            #     input_ids = input_ids.long()
            #     labels = labels.long()
            # else:
            #     buquan = max_length - input_ids.numel()
            #     awb = torch.LongTensor([eos_id for _ in range(buquan)]).view(1, buquan)
            #     input_ids = torch.cat((input_ids, awb), dim=1)
            #     input_ids = input_ids.view(1, input_ids.numel())
            #     labels = torch.cat((labels, awb), dim=1)
            #     labels = labels.view(1, labels.numel())
            #     input_ids = input_ids.long()
            #     labels = labels.long()



        else:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
                ids = example["input_ids"]
                labs = example["labels"]
                ids = ids + [pad_token_id] + [pad_token_id] * (longest - length)

                # ids = ids + [model.generation_config.pad_token_id] * (longest - length)
                labs = labs + [pad_token_id] + [-100] * (longest - length)
                # ids = ids
                # labs = labs

                input_ids.append(torch.LongTensor(ids))
                labels_list.append(torch.LongTensor(labs))

            # input_ids.append(torch.LongTensor(eos_id))
            # labels_list.append(torch.LongTensor(eos_id))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels_list)
            # eos_id = model.generation_config.eos_token_id
            # T1 = torch.LongTensor([eos_id for _ in range(len_batch)]).view(len_batch, 1)
            # a = []
            # a.append(input_ids)
            # a.append(T1)
            # b = []
            # b.append(labels)
            # b.append(T1)
            #
            # # a.append(T2)
            # # a.append(T3)
            #
            # input_ids = torch.cat(a, dim=1)
            # labels = torch.cat(b, dim=1)

        # print(input_ids)
        # print(input_ids.numel())
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    def data_collator_val(examples: list, data_strong=1, model=model_train, max_length=1024):
        len_batch = len(examples)
        len_ids = [len(example["input_ids"]) for example in examples]
        longest = max(len_ids)  # 之后按照batch中最长的input_ids进行padding

        input_ids = []
        labels_list = []
        eos_id = model.generation_config.eos_token_id
        pad_token_id = model.generation_config.pad_token_id

        # for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
        #     ids = example["input_ids"]
        #     labs = example["labels"]
        #     # ids = ids + [151643] * (longest - length)
        #
        #     #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
        #     #             labs = labs + [-100] * (longest - length)
        #     ids = ids
        #     labs = labs
        #
        #     input_ids.append(torch.LongTensor(ids))
        #
        #     labels_list.append(torch.LongTensor(labs))
        #
        # print(input_ids)

        if data_strong == 1:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            dian_dai = 1
            for example in examples:
                if dian_dai == len_batch:
                    ids = example["input_ids"]
                    labs = example["labels"]
                    # ids = ids + [151643] * (longest - length)

                    #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
                    #             labs = labs + [-100] * (longest - length)
                    ids = ids
                    labs = labs
                    len_ids = len(ids)
                    len_labs = len(labs)
                    input_ids.append(torch.LongTensor(ids).view(1, len_ids))
                    labels_list.append(torch.LongTensor(labs).view(1, len_labs))
                else:
                    ids = example["input_ids"]
                    labs = example["labels"]
                    # ids = ids + [151643] * (longest - length)

                    #             ids = ids + [model_train.generation_config.pad_token_id] * (longest - length)
                    #             labs = labs + [-100] * (longest - length)
                    len_ids = len(ids)
                    len_labs = len(labs)
                    ids = ids
                    # labs = labs
                    labs = [-100] * len_labs

                    input_ids.append(torch.LongTensor(ids).view(1, len_ids))
                    labels_list.append(torch.LongTensor(labs).view(1, len_labs))
                    dian_dai = dian_dai + 1


            input_ids = torch.cat(input_ids, dim=1)
            labels = torch.cat(labels_list, dim=1)
            len_eos = 1
            awb = torch.LongTensor([eos_id for _ in range(len_eos)]).view(1, len_eos)
            input_ids = torch.cat((input_ids, awb), dim=1)
            input_ids = input_ids.view(1, input_ids.numel())
            labels = torch.cat((labels, awb), dim=1)
            labels = labels.view(1, labels.numel())

            input_ids = input_ids.long()
            labels = labels.long()
            # numer_input = input_ids.numel()
            # numer_label = labels.numel()
            # # print("numer_input",numer_input)
            # if input_ids.numel() >= max_length:
            #     # print(input_ids)
            #     # print(input_ids[0, :max_length])
            #     input_ids = input_ids[0, :max_length]
            #     labels = labels[0, :max_length]
            #     input_ids = input_ids.view(1, input_ids.numel())
            #     labels = labels.view(1, input_ids.numel())
            #     input_ids = input_ids.long()
            #     labels = labels.long()
            # else:
            #     buquan = max_length - input_ids.numel()
            #     awb = torch.LongTensor([eos_id for _ in range(buquan)]).view(1, buquan)
            #     input_ids = torch.cat((input_ids, awb), dim=1)
            #     input_ids = input_ids.view(1, input_ids.numel())
            #     labels = torch.cat((labels, awb), dim=1)
            #     labels = labels.view(1, labels.numel())
            #     input_ids = input_ids.long()
            #     labels = labels.long()



        else:
            input_ids = []
            labels_list = []
            eos_id = model.generation_config.eos_token_id
            for length, example in sorted(zip(len_ids, examples), key=lambda x: -x[0]):
                ids = example["input_ids"]
                labs = example["labels"]
                ids = ids + [pad_token_id] + [pad_token_id] * (longest - length)

                # ids = ids + [model.generation_config.pad_token_id] * (longest - length)
                labs = labs + [pad_token_id] + [-100] * (longest - length)
                # ids = ids
                # labs = labs

                input_ids.append(torch.LongTensor(ids))
                labels_list.append(torch.LongTensor(labs))

            # input_ids.append(torch.LongTensor(eos_id))
            # labels_list.append(torch.LongTensor(eos_id))
            input_ids = torch.stack(input_ids)
            labels = torch.stack(labels_list)
            # eos_id = model.generation_config.eos_token_id
            # T1 = torch.LongTensor([eos_id for _ in range(len_batch)]).view(len_batch, 1)
            # a = []
            # a.append(input_ids)
            # a.append(T1)
            # b = []
            # b.append(labels)
            # b.append(T1)
            #
            # # a.append(T2)
            # # a.append(T3)
            #
            # input_ids = torch.cat(a, dim=1)
            # labels = torch.cat(b, dim=1)

        # print(input_ids)
        # print(input_ids.numel())
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def df_Multi(df, Multirounds):
            x2 = [x1 for x1 in range(len(df))]
            sampler = RandomSampler(data_source=x2)
            qwe = (len(df) // (Multirounds))
            batch_sampler = BatchSampler(sampler=sampler,
                                         batch_size=Multirounds, drop_last=True)
            arr = numpy.zeros((qwe, Multirounds * 2))
            b = pd.DataFrame(arr)
            a_b = [0 for _ in batch_sampler]
            for ids, ind in enumerate(batch_sampler):
                a_b[ids] = ind
            # print(a_b)
            # print(qwe)
            for i in range(qwe):
                for j in range(0, 2 * Multirounds, 2):
                    j_2 = j // 2
                    a_c = a_b[i][j_2]
                    # print(df['问题'].iloc[a_c])
                    b[j].iloc[i] = df['问题'].iloc[a_c]
                    t = df['答案'].iloc[a_c]
                    b[j + 1].iloc[i] = df[f'{t}'].iloc[a_c]
            columns_a = [f'句子{x}' for x in range(2 * Multirounds)]
            # print(columns_a)
            b.columns = columns_a
            dg = b
            return dg


    # def df_Multi(df, Multirounds):
    #     x2 = [x1 for x1 in range(len(df))]
    #     sampler = RandomSampler(data_source=x2)
    #     qwe = (len(df) // (Multirounds))
    #     batch_sampler = BatchSampler(sampler=sampler,
    #                                  batch_size=Multirounds, drop_last=True)
    #     arr = numpy.zeros((qwe, Multirounds * 2))
    #     b = pd.DataFrame(arr)
    #     a_b = [0 for _ in batch_sampler]
    #     for ids, ind in enumerate(batch_sampler):
    #         a_b[ids] = ind
    #     # print(a_b)
    #     # print(qwe)
    #     for i in range(qwe):
    #         for j in range(0, 2 * Multirounds, 2):
    #             j_2 = j // 2
    #             a_c = a_b[i][j_2]
    #             # print(df['问题'].iloc[a_c])
    #             b[j].iloc[i] = df['问题'].iloc[a_c]
    #             t = df['答案'].iloc[a_c]
    #             b[j + 1].iloc[i] = df[f'{t}'].iloc[a_c]
    #     columns_a = [f'句子{x}' for x in range(2 * Multirounds)]
    #     # print(columns_a)
    #     b.columns = columns_a
    #     dg = b
    #     return dg

    class MyDataset(Dataset):
        def __init__(self, df, yuxunlian=1
                     ):
            self.df = df
            self.yuxunlian = yuxunlian
            # self.Multirounds=Multirounds
            # self.messages = messages

        def __len__(self):
            return len(self.df)

        def get_samples(self, index):
            samples = []
            # dg=self.df_Multi(df=self.df, Multirounds=self.Multirounds)
            d = dict(self.df.iloc[index])
            samples.append(d)
            return samples

        def get_messages(self, index):
            samples = self.get_samples(index)
            # messages = deepcopy(self.messages)
            messages = []
            # print(samples)
            samples = samples[0]
            if samples['process'] != '':
                samples_gai = {'text': samples['text'], 'target': samples['process']}
            else:
                samples_gai = {'text': samples['text'], 'target': samples['answer_expressions']}

            # print(type(samples))
            # print(samples)
            for i, (d, k) in enumerate(samples_gai.items()):
                # print(d)
                if (i + 1) % 2 == 0:
                    messages.append({'role': 'assistant', 'content': k})

                # it = d['答案']
                else:
                    messages.append({'role': 'user', 'content': k})
            #print(messages)
            return messages

        def __getitem__(self, index):
            messages = self.get_messages(index)
            input_ids, labels = build_chat_input(messages, pretrain=self.yuxunlian)
            # input_ids=torch.Tensor(input_ids)
            # labels=torch.Tensor(labels)
            return {'input_ids': input_ids, 'labels': labels}

        def show_sample(self, index):
            samples = self.get_samples(index)
            # print(samples)





    # f = open("/root/experment/qwen/mmlu/data/auxiliary_train/science_middle.csv", encoding="utf-8")
    # df_train = pd.read_csv(f,names=["问题", "A", "B", "C","D","答案"])
    # # f = open("/root/experment/qwen/mmlu/data/auxiliary_train/data_sum_train.csv", encoding="utf-8")
    # # df_train = pd.read_csv(f).dropna(axis=0, how='any')
    # # message = []
    # f2 = open("/root/experment/qwen/mmlu/data/auxiliary_train/science_middle.csv", encoding="utf-8")
    # df_val = pd.read_csv(f2,names=["问题", "A", "B", "C","D","答案"])
    # # f2 = open("/root/experment/qwen/mmlu/data/data_sum_test.csv", encoding="utf-8")
    # # df_val = pd.read_csv(f2).dropna(axis=0, how='any')
    # #message = []

    data_str = open('/root/experment/llama/conic-10k/conic10k/train.json', encoding="utf-8").read()
    #print(type(data_str))
    df_train = pd.read_json(data_str, orient='records').dropna(axis=0, how='any').astype(str)
    #df.head()
    data_str_2 = open('/root/experment/llama/conic-10k/conic10k/test.json', encoding="utf-8").read()
    # print(type(data_str))
    df_val = pd.read_json(data_str_2, orient='records').dropna(axis=0, how='any').astype(str)
    # df.head()
    data_str_3 = open('/root/experment/llama/conic-10k/conic10k/dev.json', encoding="utf-8").read()
    # print(type(data_str))
    df_val_step = pd.read_json(data_str_3, orient='records').dropna(axis=0, how='any').astype(str)
    # df.head()



    from peft import get_peft_config, get_peft_model, TaskType

    model_train.supports_gradient_checkpointing = True
    model_train.gradient_checkpointing_enable()
    model_train.enable_input_require_grads()

    model_train.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    import bitsandbytes as bnb


    def find_all_linear_names(model):
        """
        找出所有全连接层，为所有全连接添加adapter
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)


    from peft import prepare_model_for_kbit_training
    model_canshu = prepare_model_for_kbit_training(model_train)

    del model_train
    for i in range(100):
       torch.cuda.empty_cache()

    lora_modules = find_all_linear_names(model_canshu)
    print(lora_modules)

    from peft import AdaLoraConfig
    peft_config = AdaLoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=64,
        lora_alpha=16, lora_dropout=0.05,
        target_modules=lora_modules
    )
    peft_model = get_peft_model(model_canshu, peft_config)

    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()

    del model_canshu
    for i in range(100):
       torch.cuda.empty_cache()



    def training_loop(epochs=40,
                      lr=1e-4,
                      ckpt_path='/root/experment/qwen/qwen_ceshi',
                      peft_model=peft_model,
                      df_train=df_train,
                      df_val=df_val,
                      accumulation_steps=1,
                      local_rank=local_rank,
                      Multirounds=4,
                      batch_size_val=1,
                      batch_size_train=4,



                      ):
        # train_dataloader, eval_dataloader = create_dataloaders(batch_size)
        # model = create_net()

        # print(ds_train[0])
        # ds_train.show_sample(0)
        # print(len(dl_train))
        # for batch in dl_train:
        #     break

        model = peft_model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                        find_unused_parameters=False, broadcast_buffers=False)
        # peft_model = nn.parallel.DistributedDataParallel(peft_model, device_ids=[local_rank],
        #                                                  find_unused_parameters=True)
        loss_function = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=5e-2)

        #df_train_Multi = df_Multi(df=df_train, Multirounds=Multirounds)

        # ds_train = MyDataset(dg)


        # ds_train = MyDataset(df_train)
        #
        # train_sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
        #
        # dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size_train, sampler=train_sampler,
        #                                        collate_fn=data_collator)

        df_train_4 = df_train.sample(frac=1)
        ds_train_4 = MyDataset(df_train_4, yuxunlian=1)
        train_sampler_4 = torch.utils.data.distributed.DistributedSampler(ds_train_4)
        dl_train_4 = torch.utils.data.DataLoader(ds_train_4, batch_size=batch_size_train, sampler=train_sampler_4,
                                                 collate_fn=data_collator_train)

        len_dl_4 = len(dl_train_4)

        df_train_3 = df_train.sample(frac=1)
        #df_train_3 = df_train_3.iloc[:len_dl_4 * 3]
        ds_train_3 = MyDataset(df_train_3, yuxunlian=1)
        train_sampler_3 = torch.utils.data.distributed.DistributedSampler(ds_train_3)
        dl_train_3 = torch.utils.data.DataLoader(ds_train_3, batch_size=3, sampler=train_sampler_3,
                                                 collate_fn=data_collator_train)

        df_train_2 = df_train.sample(frac=1)
        #df_train_2 = df_train_2.iloc[:len_dl_4 * 2]
        ds_train_2 = MyDataset(df_train_2, yuxunlian=1)
        train_sampler_2 = torch.utils.data.distributed.DistributedSampler(ds_train_2)
        dl_train_2 = torch.utils.data.DataLoader(ds_train_2, batch_size=2, sampler=train_sampler_2,
                                                 collate_fn=data_collator_train)

        df_train_1 = df_train.sample(frac=1)
        #df_train_1 = df_train_1.iloc[:len_dl_4 * 1]
        ds_train_1 = MyDataset(df_train_1, yuxunlian=1)
        train_sampler_1 = torch.utils.data.distributed.DistributedSampler(ds_train_1)
        dl_train_1 = torch.utils.data.DataLoader(ds_train_1, batch_size=1, sampler=train_sampler_1,
                                                 collate_fn=data_collator_train)


        combined_dataloader = []
        for batch_1, batch_2, batch_3, batch_4 in zip(dl_train_4, dl_train_3, dl_train_2, dl_train_1):
            combined_dataloader.append(batch_1)
            combined_dataloader.append(batch_2)
            combined_dataloader.append(batch_3)
            combined_dataloader.append(batch_4)

        dl_train = combined_dataloader
        dl_train = random.sample(dl_train, len(dl_train))

        num_training_steps = len(dl_train) * epochs
        print('num_training_steps的步骤是多少', num_training_steps)
        num_training_steps_accumulation_steps = round(num_training_steps / 16)





        # ds_train = MyDataset(df_train, yuxunlian=1)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(ds_train)
        # print(train_sampler)
        # dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size_train, sampler=train_sampler,
        #                                        collate_fn=data_collator)
        df_val_sum = df_val.sample(frac=1)

        ds_val = MyDataset(df_val_sum, yuxunlian=1)
        val_sampler_1 = torch.utils.data.distributed.DistributedSampler(ds_val)
        dl_val_1 = torch.utils.data.DataLoader(ds_val, batch_size=1, sampler=val_sampler_1,
                                               collate_fn=data_collator_val)

        ds_val = MyDataset(df_val_sum, yuxunlian=1)
        val_sampler_2 = torch.utils.data.distributed.DistributedSampler(ds_val)
        dl_val_2 = torch.utils.data.DataLoader(ds_val, batch_size=2, sampler=val_sampler_2,
                                               collate_fn=data_collator_val)

        ds_val = MyDataset(df_val_sum, yuxunlian=1)
        val_sampler_3 = torch.utils.data.distributed.DistributedSampler(ds_val)
        dl_val_3 = torch.utils.data.DataLoader(ds_val, batch_size=3, sampler=val_sampler_3,
                                               collate_fn=data_collator_val)

        ds_val = MyDataset(df_val_sum, yuxunlian=1)
        val_sampler_4 = torch.utils.data.distributed.DistributedSampler(ds_val)
        dl_val_4 = torch.utils.data.DataLoader(ds_val, batch_size=4, sampler=val_sampler_4,
                                               collate_fn=data_collator_val)

        ds_val = MyDataset(df_val_sum, yuxunlian=1)
        val_sampler_5 = torch.utils.data.distributed.DistributedSampler(ds_val)
        dl_val_5 = torch.utils.data.DataLoader(ds_val, batch_size=batch_size_val, sampler=val_sampler_5,
                                               collate_fn=data_collator_val)


        def Momentum_LR_decay(current_step=1,num_warmup_steps = 800,optimizer = optimizer,M_t = 1.0,num_training_steps=100,lr=lr):
            if current_step < num_warmup_steps:
                lr_loss = lr * float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

                xishu = progress * math.exp(-2.0 * M_t * (1 - progress))
                lr_loss = lr * max(0.0, (0.9 * 0.5 * (1.0 + math.cos(math.pi * 1.0 * xishu)) + 0.1))
            for param_group in optimizer.param_groups:
                lr_param = lr_loss
                param_group["lr"] = lr_param







        # def get_cosine_schedule_with_warmup(optimizer,
        #                                     num_training_steps, warmup_ratio=0.1, num_cycles=0.5, last_epoch=-1
        #                                     ):
        #     num_warmup_steps = 800
        #
        #     def lr_lambda(current_step):
        #         if current_step < num_warmup_steps:
        #             return float(current_step) / float(max(1, num_warmup_steps))
        #         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        #         return max(0.0,
        #                    (0.9*0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))+0.1) )
        #
        #     return LambdaLR(optimizer, lr_lambda, last_epoch)
        #
        # lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,num_training_steps=num_training_steps,warmup_ratio=0.1,
        #                                              num_cycles=0.5, last_epoch=-1
        #                                            )


        #print(optimizer.param_groups)
        # print(len(optimizer.param_groups))
        # print(type(optimizer.param_groups))
        # print(type(optimizer.param_groups[0]))
        # print(len(optimizer.param_groups[0]))
        #
        # print(optimizer.param_groups[0].keys())
        #
        # for param_group in optimizer.param_groups:
        #     lr =
        #     param_group["lr"] = lr
        #
        # for i, data in enumerate(zip(self.optimizer.param_groups, values)):
        #     param_group, lr = data
        #     param_group['lr'] = lr
        #
        # 12/0



        batch = {}


        #df_val_Multi = df_Multi(df=df_val, Multirounds=Multirounds)

        def random_input_label_SNR(batch_snr=batch, SNR=30):

            input_tensor = torch.tensor(batch_snr['input_ids'])
            label_tensor = torch.tensor(batch_snr['labels'])
            # print(input_tensor)
            a, b = input_tensor.size()
            # print(a,b)
            # print(input_tensor[0, :])
            for i in range(a):
                # print(i)
                hang_in = input_tensor[i, :]
                # print(hang_in)
                # print(len(hang_in))
                hang_la = label_tensor[i, :]
                x = [q for q in range(len(hang_in))]
                x = np.array(x)
                # print(x)
                noise = np.random.randn(len(hang_in))
                P_n = np.sum(noise ** 2) / len(noise)
                noise = noise / np.sqrt(P_n)
                P_s = np.sum(x ** 2) / len(x)
                noise = np.sqrt(P_s / (10 ** (SNR / 10))) * noise
                # print(noise)
                y2 = x + noise
                y3 = np.round(y2)
                # print(y3)
                y4 = np.abs(y3)

                # print(y4)
                stan = (len(y4) - 1)
                # print(stan)

                for j in range(len(y4)):
                    if y4[j] > stan:
                        y4[j] = stan - (y4[j] - stan)
                # print(y4)
                # print(y4)
                hang_in = hang_in[y4]
                # print(hang_in)


                input_tensor[i, :] = hang_in
                label_tensor[i, :] = hang_la

                # print(input_tensor)
                # print(label_tensor)

            batch_snr['input_ids'] = input_tensor.long()
            batch_snr['labels'] = label_tensor.long()
            #print(type(batch['input_ids']))
            batch_out = batch_snr
            return batch_out



        train_epoch_excl = []
        test_epoch_excl = []
        df_1 = pd.DataFrame(columns=['step', 'train Loss'])  # 列名
        df_1.to_csv(ckpt_path + f'/train_{local_rank}.csv', index=False)  # 路径可以根据需要更改
        df_2 = pd.DataFrame(columns=['step', 'val Loss'])  # 列名
        df_2.to_csv(ckpt_path + f'/0_shot_val_{local_rank}.csv', index=False)  # 路径可以根据需要更改
        df_2.to_csv(ckpt_path + f'/1_shot_val_{local_rank}.csv', index=False)
        df_2.to_csv(ckpt_path + f'/2_shot_val_{local_rank}.csv', index=False)
        df_2.to_csv(ckpt_path + f'/3_shot_val_{local_rank}.csv', index=False)  # 路径可以根据需要更改
        df_2.to_csv(ckpt_path + f'/4_shot_val_{local_rank}.csv', index=False)

        #torch.autograd.set_detect_anomaly = True
        lr_loss = 0
        count_loss = 0
        M_t = 0
        for epoch in range(epochs):

            df_train_1 = df_train.sample(frac=1)
            # df_train_1 = df_train_1.iloc[:len_dl_4 * 1]
            ds_train_1 = MyDataset(df_train_1, yuxunlian=1)
            train_sampler_1 = torch.utils.data.distributed.DistributedSampler(ds_train_1)
            dl_train_1 = torch.utils.data.DataLoader(ds_train_1, batch_size=1, sampler=train_sampler_1,
                                                     collate_fn=data_collator_train)

            combined_dataloader = []
            for batch_1 in dl_train_1:
                combined_dataloader.append(batch_1)

            zen_jia = random.sample(combined_dataloader, 3)

            for batch_zeng_jia in zen_jia:
                combined_dataloader.append(batch_zeng_jia)

            dl_train = combined_dataloader
            dl_train = random.sample(dl_train, len(dl_train))

            num_training_steps_epoch = len(dl_train)
            print('num_training_steps的步骤是多少', num_training_steps_epoch)



            for batch in dl_train:
                break


            model.train()


            #dl_train.sampler.set_epoch(epoch)
            #optimizer.zero_grad()

            random.seed(3407 + local_rank + epoch)
            dl_train = random.sample(dl_train, len(dl_train))
            loss_ti = 0


            loss_2 = 0
            #loss_2_inplace = 0
            for step, batch in enumerate(dl_train):
                # features,labels = batch
                #             batch['input_ids']=batch['input_ids']
                #             batch['labels']=batch['labels']
                # print(type(batch['labels']))
                my_context = model.no_sync if local_rank != -1 and (step + 1) % accumulation_steps != 0 else nullcontext
                with my_context():
                    # if step % 10 == 0:
                    #     del loss_2
                    #     for you in range(10):
                    #         torch.cuda.empty_cache()
                    #     loss_2 = 0
                    batch_1 = batch
                    batch_snr = batch
                    #batch_1 = random_input_label_SNR(batch=batch, SNR=100)


                    batch_1['input_ids'] = batch_1['input_ids'].to(device)
                    batch_1['labels'] = batch_1['labels'].to(device)
                    #print(batch_1['input_ids'])
                    #batch=batch.to(device)
                    out1 = model.forward(**batch_1)
                    #loss_ti = loss_ti + out.loss
                    #print(f'在{step}步loss_train的值{out.loss}')
                    lm_logits1 = out1.logits
                    shift_logits1 = lm_logits1[..., :-1, :].contiguous()
                    sb1 = batch_1['labels'].to(device)
                    shift_labels1 = sb1[..., 1:].contiguous()
                    # loss_tz = loss_function(
                    #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    # )
                    output1 = shift_logits1.view(-1, shift_logits1.size(-1))
                    label1 = shift_labels1.view(-1)
                    #output1_1 = output1
                    #label1_1 = label1
                    loss_1 = loss_function(
                        output1, label1
                    )
                    loss = loss_1
                    step_epoch = epoch * len(dl_train) + step
                    list_1 = [step_epoch, loss.item()]
                    # print(f'在{step}步loss的值{loss_tz}')
                    loss_ti = loss_ti + loss.item()
                    loss = loss / accumulation_steps
                    # loss_1.backward(retain_graph=True)
                    lr_loss = lr_loss+loss.item()
                    loss.backward()

                    del loss, loss_1
                    for yui in range(10):
                        torch.cuda.empty_cache()



                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count_loss = count_loss + 1
                    M_t = float(0.9*M_t + 0.1*lr_loss)
                    Momentum_LR_decay(current_step=count_loss, num_warmup_steps=50, optimizer=optimizer, M_t=M_t,
                                      num_training_steps=num_training_steps_accumulation_steps, lr=lr)
                    # print(optimizer.state_dict()['param_groups'][0]['lr'])
                    lr_loss = 0

                # lr_scheduler.step()
                acc_step = math.floor(num_training_steps_epoch / 4)
                if (step + 1) % acc_step == 0:
                    ds_val = MyDataset(df_val_step, yuxunlian=1)
                    val_sampler_1 = torch.utils.data.distributed.DistributedSampler(ds_val)
                    dl_val_1 = torch.utils.data.DataLoader(ds_val, batch_size=1, sampler=val_sampler_1,
                                                           collate_fn=data_collator_val)
                    model.eval()
                    with torch.no_grad():
                        dl_val_1.sampler.set_epoch(epoch)
                        eval_loss = 0
                        prepxi_sum = 0
                        for step, batch in enumerate(dl_val_1):
                            # features,labels = batch
                            batch['input_ids'] = batch['input_ids'].to(device)
                            batch['labels'] = batch['labels'].to(device)
                            loss_e = model.forward(**batch)
                            lm_logits = loss_e.logits
                            shift_logits = lm_logits[..., :-1, :].contiguous()
                            sb = batch['labels'].to(device)
                            shift_labels = sb[..., 1:].contiguous()
                            loss_ez = loss_function(
                                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                            )
                            # step_epoch = epoch * len(dl_val_1) + step
                            # list_2 = [step_epoch, loss_ez.item()]
                            # data = pd.DataFrame([list_2])
                            # data.to_csv(ckpt_path + f'/0_shot_val_{local_rank}.csv', mode='a', header=False,
                            #             index=False)
                            eval_loss = eval_loss + loss_ez.item()
                            prepxi = math.pow(2, loss_ez.item())
                            prepxi_sum = prepxi_sum + prepxi

                            # print(f'在{step}步loss_ez的值{loss_ez}')
                    eval_losses = eval_loss / (len(dl_val_1))
                    prepxi_sum = prepxi_sum / (len(dl_val_1))
                    print(
                        f'every_step,在0-shot任务中，,平均{len(dl_val_1)}个样本上的Loss: {eval_losses}')
                    print(f'every_step,在0-shot任务中,在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)
                    model.train()



                data = pd.DataFrame([list_1])
                data.to_csv(ckpt_path + f'/train_{local_rank}.csv', mode='a', header=False, index=False)
                #print(optimizer.state_dict()['param_groups'][0]['lr'])
                #print(lr_scheduler)


            loss_train_es = loss_ti / len(dl_train)


            print(f'在第{local_rank}号显卡上在第{epoch + 1}轮上整体训练集,平均{len(dl_train)}个样本上的Loss: {loss_train_es}')
            # train_epoch_excl.append(loss_train_es)
            # Loss_train_excel = torch.tensor(train_epoch_excl)
            # torch.save(Loss_train_excel, ckpt_path + '/Loss_train_excel/epoch_{}'.format(epoch + 1))

            df_val_sum = df_val.sample(frac=1)

            ds_val = MyDataset(df_val_sum, yuxunlian=1)
            val_sampler_1 = torch.utils.data.distributed.DistributedSampler(ds_val)
            dl_val_1 = torch.utils.data.DataLoader(ds_val, batch_size=1, sampler=val_sampler_1,
                                                   collate_fn=data_collator_val)

            ds_val = MyDataset(df_val_sum, yuxunlian=1)
            val_sampler_2 = torch.utils.data.distributed.DistributedSampler(ds_val)
            dl_val_2 = torch.utils.data.DataLoader(ds_val, batch_size=2, sampler=val_sampler_2,
                                                   collate_fn=data_collator_val)

            ds_val = MyDataset(df_val_sum, yuxunlian=1)
            val_sampler_3 = torch.utils.data.distributed.DistributedSampler(ds_val)
            dl_val_3 = torch.utils.data.DataLoader(ds_val, batch_size=3, sampler=val_sampler_3,
                                                   collate_fn=data_collator_val)

            ds_val = MyDataset(df_val_sum, yuxunlian=1)
            val_sampler_4 = torch.utils.data.distributed.DistributedSampler(ds_val)
            dl_val_4 = torch.utils.data.DataLoader(ds_val, batch_size=4, sampler=val_sampler_4,
                                                   collate_fn=data_collator_val)

            ds_val = MyDataset(df_val_sum, yuxunlian=1)
            val_sampler_5 = torch.utils.data.distributed.DistributedSampler(ds_val)
            dl_val_5 = torch.utils.data.DataLoader(ds_val, batch_size=batch_size_val, sampler=val_sampler_5,
                                                   collate_fn=data_collator_val)

            model.eval()
            with torch.no_grad():
                dl_val_1.sampler.set_epoch(epoch)
                eval_loss = 0
                prepxi_sum = 0
                for step, batch in enumerate(dl_val_1):
                    # features,labels = batch
                    batch['input_ids'] = batch['input_ids'].to(device)
                    batch['labels'] = batch['labels'].to(device)
                    loss_e = model.forward(**batch)
                    lm_logits = loss_e.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    sb = batch['labels'].to(device)
                    shift_labels = sb[..., 1:].contiguous()
                    loss_ez = loss_function(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    step_epoch = epoch * len(dl_val_1) + step
                    list_2 = [step_epoch, loss_ez.item()]
                    data = pd.DataFrame([list_2])
                    data.to_csv(ckpt_path + f'/0_shot_val_{local_rank}.csv', mode='a', header=False, index=False)
                    eval_loss = eval_loss + loss_ez.item()
                    prepxi = math.pow(2, loss_ez.item())
                    prepxi_sum = prepxi_sum + prepxi

                    # print(f'在{step}步loss_ez的值{loss_ez}')
            eval_losses = eval_loss / (len(dl_val_1))
            prepxi_sum = prepxi_sum / (len(dl_val_1))
            print(
                f'在0-shot任务中，第{local_rank}号显卡上第{epoch + 1}轮上整体验证集,平均{len(dl_val_1)}个样本上的Loss: {eval_losses}')
            print(f'在0-shot任务中,在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)

            with torch.no_grad():
                dl_val_2.sampler.set_epoch(epoch)
                eval_loss = 0
                prepxi_sum = 0
                for step, batch in enumerate(dl_val_2):
                    # features,labels = batch
                    batch['input_ids'] = batch['input_ids'].to(device)
                    batch['labels'] = batch['labels'].to(device)
                    loss_e = model.forward(**batch)
                    lm_logits = loss_e.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    sb = batch['labels'].to(device)
                    shift_labels = sb[..., 1:].contiguous()
                    loss_ez = loss_function(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    step_epoch = epoch * len(dl_val_2) + step
                    list_2 = [step_epoch, loss_ez.item()]
                    data = pd.DataFrame([list_2])
                    data.to_csv(ckpt_path + f'/1_shot_val_{local_rank}.csv', mode='a', header=False, index=False)
                    eval_loss = eval_loss + loss_ez.item()
                    prepxi = math.pow(2, loss_ez.item())
                    prepxi_sum = prepxi_sum + prepxi

                    # print(f'在{step}步loss_ez的值{loss_ez}')
            eval_losses = eval_loss / (len(dl_val_2))
            prepxi_sum = prepxi_sum / (len(dl_val_2))
            print(
                f'在1-shot任务中，第{local_rank}号显卡上第{epoch + 1}轮上整体验证集,平均{len(dl_val_2)}个样本上的Loss: {eval_losses}')
            print(f'在1-shot任务中,在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)

            with torch.no_grad():
                dl_val_3.sampler.set_epoch(epoch)
                eval_loss = 0
                prepxi_sum = 0
                for step, batch in enumerate(dl_val_3):
                    # features,labels = batch
                    batch['input_ids'] = batch['input_ids'].to(device)
                    batch['labels'] = batch['labels'].to(device)
                    loss_e = model.forward(**batch)
                    lm_logits = loss_e.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    sb = batch['labels'].to(device)
                    shift_labels = sb[..., 1:].contiguous()
                    loss_ez = loss_function(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    step_epoch = epoch * len(dl_val_3) + step
                    list_2 = [step_epoch, loss_ez.item()]
                    data = pd.DataFrame([list_2])
                    data.to_csv(ckpt_path + f'/2_shot_val_{local_rank}.csv', mode='a', header=False, index=False)
                    eval_loss = eval_loss + loss_ez.item()
                    prepxi = math.pow(2, loss_ez.item())
                    prepxi_sum = prepxi_sum + prepxi

                    # print(f'在{step}步loss_ez的值{loss_ez}')
            eval_losses = eval_loss / (len(dl_val_3))
            prepxi_sum = prepxi_sum / (len(dl_val_3))
            print(
                f'在2-shot任务中，第{local_rank}号显卡上第{epoch + 1}轮上整体验证集,平均{len(dl_val_3)}个样本上的Loss: {eval_losses}')
            print(f'在2-shot任务中,在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)


            with torch.no_grad():
                dl_val_4.sampler.set_epoch(epoch)
                eval_loss = 0
                prepxi_sum = 0
                for step, batch in enumerate(dl_val_4):
                    # features,labels = batch
                    batch['input_ids'] = batch['input_ids'].to(device)
                    batch['labels'] = batch['labels'].to(device)
                    loss_e = model.forward(**batch)
                    lm_logits = loss_e.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    sb = batch['labels'].to(device)
                    shift_labels = sb[..., 1:].contiguous()
                    loss_ez = loss_function(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    step_epoch = epoch * len(dl_val_4) + step
                    list_2 = [step_epoch, loss_ez.item()]
                    data = pd.DataFrame([list_2])
                    data.to_csv(ckpt_path + f'/3_shot_val_{local_rank}.csv', mode='a', header=False, index=False)
                    eval_loss = eval_loss + loss_ez.item()
                    prepxi = math.pow(2, loss_ez.item())
                    prepxi_sum = prepxi_sum + prepxi

                    # print(f'在{step}步loss_ez的值{loss_ez}')
            eval_losses = eval_loss / (len(dl_val_4))
            prepxi_sum = prepxi_sum / (len(dl_val_4))
            print(
                f'在3-shot任务中，第{local_rank}号显卡上第{epoch + 1}轮上整体验证集,平均{len(dl_val_4)}个样本上的Loss: {eval_losses}')
            print(f'在3-shot任务中，在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)

            with torch.no_grad():
                dl_val_5.sampler.set_epoch(epoch)
                eval_loss = 0
                prepxi_sum = 0
                for step, batch in enumerate(dl_val_5):
                    # features,labels = batch
                    batch['input_ids'] = batch['input_ids'].to(device)
                    batch['labels'] = batch['labels'].to(device)
                    loss_e = model.forward(**batch)
                    lm_logits = loss_e.logits
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    sb = batch['labels'].to(device)
                    shift_labels = sb[..., 1:].contiguous()
                    loss_ez = loss_function(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                    step_epoch = epoch * len(dl_val_5) + step
                    list_2 = [step_epoch, loss_ez.item()]
                    data = pd.DataFrame([list_2])
                    data.to_csv(ckpt_path + f'/4_shot_val_{local_rank}.csv', mode='a', header=False, index=False)
                    eval_loss = eval_loss + loss_ez.item()
                    prepxi = math.pow(2, loss_ez.item())
                    prepxi_sum = prepxi_sum + prepxi

                    # print(f'在{step}步loss_ez的值{loss_ez}')
            eval_losses = eval_loss / (len(dl_val_5))
            prepxi_sum = prepxi_sum / (len(dl_val_5))
            print(
                f'在4-shot任务中，第{local_rank}号显卡上第{epoch + 1}轮上整体验证集,平均{len(dl_val_5)}个样本上的Loss: {eval_losses}')
            print(f'在4-shot任务中,在第{epoch + 1}轮上整体训练集的每个样本困惑度为', prepxi_sum)

            if dist.get_rank() == 0:
                model.module.save_pretrained(ckpt_path)
        # accelerator.wait_for_everyone()
        # unwrap_net = accelerator.unwrap_model(model)
        #f.close()





    # device=torch.device("cuda",local_rank)


    ckpt_path_qwen = '/root/experment/ICIC/QWEN/ceshi/Baichuan_01_02'
    training_loop(epochs=2,
                  lr=1e-4,
                  ckpt_path=ckpt_path_qwen,
                  peft_model=peft_model,
                  df_train=df_train,
                  df_val=df_val,
                  accumulation_steps=16,
                  local_rank=local_rank,
                  Multirounds=1,
                  batch_size_train=4,
                  batch_size_val=5,
                  )