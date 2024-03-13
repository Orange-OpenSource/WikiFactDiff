#### WHY CLEAN? look at this example :
# Jack Dorsey, position held:
# chief executive officer (('forget', 'keep'))
####


from collections import Counter
import random
from build.config import STORAGE_FOLDER
import os.path as osp
import json
import numpy as np

random.seed(456)


if __name__ == '__main__':
    wfd = [json.loads(x) for x in open(osp.join(STORAGE_FOLDER, 'wikifactdiff_dirty.jsonl'))]
    n1,n2,n3 = 0,0,0
    for x in wfd:
        count_objects = Counter([y['label'] for y in x['objects']])
        
        if len(count_objects) == len(x['objects']):
            continue
        new_objects = []
        for k,v in count_objects.items():
            if v == 1:
                new_objects.extend([y for y in x['objects'] if y['label'] == k])
                continue
            labels, decisions = list(zip(*[(y['label'], y['decision']) for y in x['objects'] if y['label'] == k]))
            count_decisions = Counter(decisions)
            n_learn = count_decisions.get('learn', 0)
            n_forget = count_decisions.get('forget', 0)
            n_keep = count_decisions.get('keep', 0)
            if n_learn == 0 and n_forget > 0 and n_keep == 0:
                new_objects.append([y for y in x['objects'] if y['label'] == k][0])
                n1 += 1
            elif n_learn > 0 and n_forget == 0 and n_keep >= 0:
                new_objects.append([y for y in x['objects'] if y['label'] == k][0])
                n2 += 1
            elif n_learn == 0 and n_forget == 0 and n_keep > 0:
                new_objects.append([y for y in x['objects'] if y['label'] == k][0])
                n3 += 1
            else:
                new_objects.clear()
                break
        x['objects'] = new_objects
    wfd = [x for x in wfd if len(x['objects'])]

    # Rename labels
    rename = {
        'learn' : 'new',
        'forget' : 'obsolete',
        'keep' : "static"
    }

    for x in wfd:
        for obj in x['objects']:
            obj['decision'] = rename[obj['decision']]

    with open(osp.join(STORAGE_FOLDER, 'wikifactdiff.jsonl'), 'w') as f:
        for x in wfd:
            f.write(json.dumps(x) + '\n')
    # wfd_repl = [x for x in wfd if x['is_replace'] and (random.random() < 1/14 or x['relation']['label'] != 'population')]
    # with open(osp.join(STORAGE_FOLDER, 'wikifactdiff_replacement.jsonl'), 'w') as f:
    #     for x in wfd_repl:
    #         f.write(json.dumps(x) + '\n')

    
    print('Filter process statistics:')
    print('Only forgets : %s' % n1)
    print('Only learns and keeps : %s' % n2)
    print('Only keeps : %s' % n3)