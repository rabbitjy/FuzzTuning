import os
import json
from tqdm import tqdm
import time
import functools
import signal
from transformers import RobertaTokenizer

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

def files(path):
    g = os.walk(path) 
    file=[]
    for path,dir_list,file_list in g:  
        for file_name in file_list:  
            new_file = os.path.join(path, file_name)
            file.append(new_file)
    return file

cont=0
with open("train_fuzz_clone.jsonl", 'w') as f:
    for i in tqdm(range(1,65),total=64):
        items=files("ProgramData/{}".format(i))
        for item in items:
            # print('item is :', item)
            js={}
            js['label']=item.split('/')[1]
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
            
            f.write(json.dumps(js)+'\n')
            cont+=1
        
with open("valid_fuzz_clone.jsonl", 'w') as f:
    for i in tqdm(range(65,81),total=16):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
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
            
            f.write(json.dumps(js)+'\n')
            cont+=1
            
with open("test_fuzz_clone.jsonl", 'w') as f:
    for i in tqdm(range(81,105),total=24):
        items=files("ProgramData/{}".format(i))
        for item in items:
            js={}
            js['label']=item.split('/')[1]
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
            
            f.write(json.dumps(js)+'\n')
            cont+=1
