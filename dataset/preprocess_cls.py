import os
import sys
import re
import random
import json
from tqdm import tqdm
import time
import functools
import signal
from transformers import RobertaTokenizer

# settings
seed = 0
in_dir = './ProgramData/'
num_classes = 104
random.seed(seed)

fuzz_num = 0
max_fuzz_len = 64
prompt_len = 7
tokenizer = RobertaTokenizer.from_pretrained('../microsoft/codebert-base', do_lower_case=False, cache_dir="")
block_size_source = 400
n_max = 32
limit_time = 3
large_num = 10000

def timeout(sec):
    """
    timeout decorator
    :param sec: function raise TimeoutError after ? seconds
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):

            def _handle_timeout(signum, frame):
                err_msg = f'Function {func.__name__} timed out after {sec} seconds'
                raise TimeoutError(err_msg)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(sec)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped_func
    return decorator

@timeout(limit_time)
def read_fuzz_file(path):
    input = open(path, 'r', encoding='utf-8').read()
    return input

def is_illegal_str(text):
    if (text is not None) and (len(text) > 0) and (len(text) < large_num):
        return True
    return False


# file
f_train = open("train_fuzz_cls.jsonl", 'w')
f_valid = open("valid_fuzz_cls.jsonl", 'w')
f_test = open("test_fuzz_cls.jsonl", 'w')

# process
cont = 0
for i in tqdm(range(1, num_classes + 1)):
    for f in sorted(os.listdir(os.path.join(in_dir, str(i)))):
        in_filename = os.path.join(in_dir, str(i), f)
        r = random.random()
        if r < 0.6:
            split = 'train'
        elif r < 0.8:
            split = 'valid'
        else:
            split = 'test'
        
        # fuzz info
        item = in_filename
        js={}
        js['label']=item.split('/')[-2]
        js['index']=str(cont)
        js['code']=open(item,encoding='latin-1').read()
        js['fuzz'] = []

        code=' '.join(js['code'].split())
        code_tokens = tokenizer.tokenize(code)[:block_size_source-2]
        source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]

        '''start add fuzz input and output'''
        fuzz_path = './output/' + str(i) + '/' + item.split('/')[-1].split('.')[0] + '/' + 'default/'
        if os.path.exists(fuzz_path):
            input_path = fuzz_path + 'queue/'
            output_path = fuzz_path + 'output/'
            for file in os.listdir(input_path):
                if 'id' not in file:
                    continue

                input = None
                output = None

                try:
                    input = read_fuzz_file(input_path+file)
                    if not is_illegal_str(input):
                        input = None
                except Exception:
                    input = None

                try:
                    output = read_fuzz_file(output_path+file)
                    if not is_illegal_str(output):
                        output = None
                except Exception:
                    output = None

                # only add useful input-output pairs
                if (input is not None) and (output is not None):
                    total_len = len(tokenizer.tokenize(input)) + len(tokenizer.tokenize(output)) + prompt_len
                    if total_len <= max_fuzz_len:
                        js['fuzz'].append((input, output))
                        if len(js['fuzz']) >= n_max:
                           break

        if (len(js['fuzz']) < fuzz_num):
            continue
        '''end add'''
        
        if split == 'train':
            f_train.write(json.dumps(js)+'\n')
        elif split == 'valid':
            f_valid.write(json.dumps(js)+'\n')
        elif split == 'test':
            f_test.write(json.dumps(js)+'\n')
        else:
            raise NotImplementedError
        cont+=1
        
f_train.close()
f_valid.close()
f_test.close()
