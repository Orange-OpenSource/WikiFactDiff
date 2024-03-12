from collections import defaultdict
from multiprocessing import Process, Queue, set_start_method
import torch
import numpy as np
import queue
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import h5py
import tqdm
from build.find_neighbors.config import EMBEDDINGS_FOLDER
import os.path as osp
from build.utils.wd import db
import argparse
import json
import shutil
import os
import re
import gc

import subprocess as sp

MODEL_NAME = "EleutherAI/gpt-neo-2.7B"
BATCH_SIZE = 256
RESUME = "yes"
VERSION = "old"
MM_GPU_FILTER = 23000

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class Finish:
    pass



num_gpus = torch.cuda.device_count()

def gpu_worker(subfolder_path, gpu_id, model_name : str, task_queue : Queue, start_index : int):
    torch.cuda.set_device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to('cuda')
    emb_size = model.get_input_embeddings().embedding_dim


    tokenizer.pad_token = tokenizer.eos_token
    file = open(osp.join(subfolder_path, f'encoded_sentences_{gpu_id}.jsonl'), 'a')
    end_index = None

    with h5py.File(osp.join(subfolder_path, f'encoded_sentences_{gpu_id}.h5'), 'a') as f:
        if "encodings" not in f.keys():
            dset = f.create_dataset("encodings", (100000, emb_size), maxshape=(None, emb_size), dtype='float16')
        else:
            dset = f['encodings']
        
        while True:
            task = task_queue.get()
            if task is Finish:
                break

            # Tokenize the sentences
            inputs = tokenizer([x['label'] for x in task], padding=True, truncation=True, return_tensors="pt", max_length=50)
            
            # Move to GPU
            inputs = {k: v.to(gpu_id) for k, v in inputs.items()}
            
            # Run through model
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # The output embeddings (for all tokens)
            all_hidden_states = outputs.hidden_states[-1]

            # Use the attention mask to find the last token of each sentence
            attention_mask = inputs["attention_mask"]
            last_token_positions = attention_mask.sum(dim=1) - 1  # -1 for 0-based indexing

            # Extract the embeddings of the last token in each sentence
            embeddings = all_hidden_states[range(len(task)), last_token_positions].cpu().numpy()
            
            # Write to disk
            end_index = start_index + len(task)

            if dset.shape[0] < end_index :
                print(f'insufficient dset size, end={end_index}; resizing')
                dset.resize(end_index + end_index // 2, axis=0)
            dset[start_index:end_index] = embeddings
            file.write(''.join(json.dumps(t) + '\n' for t in task))
            file.flush()

            start_index = end_index

            gc.collect()
        if end_index is not None:
            dset.resize(end_index, axis=0)



def label_generator(version, batch_size, seen_entities : set):
    l = []
    n = db['wikidata_%s_prep' % version].estimated_document_count()
    for x in tqdm.tqdm(db['wikidata_%s_prep' % version].find({'label' : {'$exists' : 1}}, {'label' : 1}), total=n):
        if x['_id'] in seen_entities:
            continue
        l.append(x)
        if len(l) % batch_size == 0:
            yield l
            l = []


def main(subfolder_name : str, model_name=MODEL_NAME, batch_size=BATCH_SIZE, resume=RESUME, version=VERSION, mm_gpu_filter=MM_GPU_FILTER):
    subfolder_path = osp.join(EMBEDDINGS_FOLDER, subfolder_name)
    seen_entities = set()
    int2start = defaultdict(lambda : 0)
    if resume == 'no':
        shutil.rmtree(subfolder_path, ignore_errors=True)
        os.makedirs(subfolder_path, exist_ok=True)
    else:
        os.makedirs(subfolder_path, exist_ok=True)
        for file in os.listdir(subfolder_path):
            if m := re.match(r"^encoded_sentences_([0-9]+)\.jsonl$", file):
                with open(osp.join(subfolder_path, file)) as f:
                    jsonl = [json.loads(x) for x in f.readlines()]
                    seen_entities.update([x['_id'] for x in jsonl])
                    int2start[int(m.group(1))] = len(jsonl)

                assert osp.exists(osp.join(subfolder_path, 'encoded_sentences_%s.h5' % m.group(1)))
    # try:
    #     set_start_method('spawn')
    # except RuntimeError:
    #     pass
    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    print('Folder where to save embeddings : %s' % subfolder_path)
    print('Resume : %s' % resume)
    print('Filter all GPUs with less memory than %s MBs' % mm_gpu_filter)

    task_queue = Queue(100)
    gpu_available_memory = get_gpu_memory()
    ps = [Process(target=gpu_worker, args=(subfolder_path, i, model_name, task_queue, int2start[i])) for i in range(num_gpus) if gpu_available_memory[i] > mm_gpu_filter]
    for p in ps:
        p.start()
        
    for batch in label_generator(version, batch_size, seen_entities):
        task_queue.put(batch)
    for _ in range(num_gpus):
        task_queue.put(Finish)
    for p in ps:
        p.join()
    print('Finished!')
    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for encoding sentences with Large Language Models from huggingface.")


    parser.add_argument("--subfolder_name", type=str, required=True,
                        help="Name of the subfolder in the EMBEDDINGS_FOLDER (defined in find_neighbors/config.py) where to store embeddings.")

    # Add argument for specifying the model's name
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="Name of the model from huggingface to use. Default is '%s'." % MODEL_NAME)

    # Add argument for specifying the batch size
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for encoding. Default is %s." % BATCH_SIZE)
    
    parser.add_argument('--resume', type=str, default=RESUME, help='Attempt to resume the encoding. Can take three values : "no" do not attempt a resume, "attempt" will attempt a resume (DATA VALIDITY NOT GUARANTEED). Defaults to "%s"' % RESUME, 
                        choices=['no', 'yes'])
    
    parser.add_argument('--version', type=str, default=VERSION, help='From which wikidata version to retrieve labels. Defaults to "%s"' % VERSION, 
                        choices=['old', 'new'])
    parser.add_argument('--mm_gpu_filter', type=int, default=MM_GPU_FILTER, help="Do not use GPUs that have less memory than this argument (in MBs). Defaults to %s" % MM_GPU_FILTER)
    args = parser.parse_args()
    main(args.subfolder_name, args.model_name, args.batch_size, args.resume, args.version, args.mm_gpu_filter)
