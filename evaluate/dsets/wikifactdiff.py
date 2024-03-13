import json
import random
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset
from collections import Counter

from util import globals

REMOTE_ROOT = f"{globals.REMOTE_ROOT_URL}/data/dsets"
WIKIFACTDIFF_PATH = "/wfd_storage/wikifactdiff.jsonl"

class WikiFactDiffDataset(Dataset):
    def __init__(
        self,
        functional_only: bool = False,
        size: typing.Optional[int] = None,
        balance : bool = True,
        path = None,
        use_random_neighbors = False,
        *args,
        **kwargs,
    ):
        if path is None:
            path = WIKIFACTDIFF_PATH
        self.functional_only = functional_only
        with open(path, "r") as f:
            data = [(i,json.loads(x)) for i,x in enumerate(f.readlines())]
            for i, x in data:
                x['case_id'] = i
            self.data = list(zip(*data))[1]
        if use_random_neighbors:
            random.seed(456465)
            print('Using random neighbors to compute specificity.')
            all_neighborhood = []
            for x in self.data:
                all_neighborhood.extend(x['neighborhood'])
            for x in self.data:
                x['neighborhood'] = random.sample(all_neighborhood, k=10)
        if functional_only:
            print('Keep only replace updates.')
            self.data = [x for x in self.data if x['is_replace']]

        # undersample_population : portion to keep from updates on the relation "population" 
        if balance and functional_only:
            random.seed(78451)
            print('Undersample updates on the "population" relation by a factor of 14')
            self.data = [x for x in self.data if x['relation']['label'] != 'population' or random.random() < 1/14]
        self.balance = balance

        if size is not None:
            self.data = self.data[:size]


        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
class FunctionalWikiFactDiffDataset(WikiFactDiffDataset):
    @staticmethod
    def adapt_instance(instance : dict) -> dict:
        new_update, old_update = sorted(instance['objects'], key=lambda x : x['decision'])
        to_d = lambda x : {'id': x.get('id'), 'str' : x['label']}
        d = {'subject' : instance['subject']['label'], 'relation_id' : instance['relation']['id'], 
             'target_true' : to_d(old_update), 'target_new' : to_d(new_update),
             "prompt" : instance['update_prompt'].replace(' ____', '').replace(instance['subject']['label'], '{}')}
        d = {'requested_rewrite' : d}
        paraphrase_prompts = [x.replace(' ____', '') for x in instance['generalization_prompts']]
        neighborhood_prompts = [{'prompt' : v['objects'][0]['prompt'].replace(' ____', ''), 'expected_object' : v['objects'][0]['object']['label']} for v in instance['neighborhood']]
        d['neighborhood_prompts'] = neighborhood_prompts
        d['paraphrase_prompts'] = paraphrase_prompts
        d['case_id'] = instance['case_id']
        return d

    def __init__(self, data_dir = None, size : int = None, adapt_to_cf_format=True, *args, **kwargs):
        # undersample_population : portion to keep from updates on the relation "population" 
        super().__init__(True, size, balance = True, *args, **kwargs)
        if adapt_to_cf_format:
            self.data = [FunctionalWikiFactDiffDataset.adapt_instance(x) for x in self.data]


        # # For debug purposes
        # for idx, x in enumerate(self.data):
        #     x = x['requested_rewrite']
        #     if x['subject'] == 'JKN Global Group' and x['relation_id'] == 'P2561':
        #         break
        # else:
        #     idx = None
        # if idx is not None:
        #     self.data = self.data[idx:]


if __name__ == '__main__':
    d = FunctionalWikiFactDiffDataset()
    print(d)
