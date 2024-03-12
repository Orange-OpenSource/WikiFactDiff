# Compute diff and detect physically new entities

from __future__ import annotations

from collections import defaultdict
import json

import numpy as np
from build.config import STORAGE_FOLDER
from build.utils.ku import correct_time
from build.utils.wd import db, sim_dict
import os.path as osp
from copy import deepcopy
from build.utils.ku import old_wikidata_timestamp

_DEBUG_NUM_ENT = -1

class GlobalStat:
    n_new_entities = 0
    n_both_entities = 0
    n_old_entities = 0

    n_new_triples = 0
    n_both_triples = 0
    n_old_triples = 0

    n_new_group = 0
    n_both_group = 0
    n_old_group = 0

    n_fully_determined = 0
    n_partially_determined = 0
    n_not_determined = 0

    physicality_count = defaultdict(lambda : 0)

    @staticmethod
    def diff_analysis(diff : dict):
        n_triples = 0
        n_new, n_both, n_old = 0,0,0
        for _, snaks in diff['diff'].items():
            n_new_group, n_both_group, n_old_group = 0, 0, 0
            for snak in snaks:
                flag = snak['diff_flag']
                n_triples += 1
                if flag == 'new':
                    n_new += 1
                    n_new_group += 1
                elif flag == 'both':
                    n_both += 1
                    n_both_group += 1
                elif flag == 'old':
                    n_old += 1
                    n_old_group += 1
                else:
                    assert False
                
            if n_new_group == len(snaks):
                GlobalStat.n_new_group += 1
            elif n_both_group == len(snaks):
                GlobalStat.n_both_group += 1
            elif n_old_group == len(snaks):
                GlobalStat.n_old_group += 1
        
        if n_new == n_triples:
            GlobalStat.n_new_entities += 1
        elif n_old == n_triples:
            GlobalStat.n_old_entities += 1
        elif n_both == n_triples:
            GlobalStat.n_both_entities += 1
        
        GlobalStat.n_old_triples += n_old
        GlobalStat.n_new_triples += n_new
        GlobalStat.n_both_triples += n_both

    @staticmethod
    def save(path : str):
        d = {k : (dict(v) if isinstance(v,defaultdict) else v) for k,v in vars(GlobalStat).items() if isinstance(v, (int, defaultdict))}
        with open(path, 'w') as f:
            json.dump(d, f, indent=4)

new_dateproperty2text = dict(P571 = dict(name='inception'), 
                             P569 = dict(name='date of birth'),
                             P580 = dict(name='start time'), 
                             P575 = dict(name='time of discovery or invention'), 
                             P1619 = dict(name='date of official opening'), 
                             P6949 = dict(name='announcement date'),
                             P585 = dict(name='point in time'))
new_dateproperty_mult2text = dict(P577 = dict(name= 'publication date', reduce="oldest"))

def new_in_wikidata_categorize(entity : dict):
        found = None
        for k in entity:
            if (k in new_dateproperty2text or k in new_dateproperty_mult2text):
                found = k
                break
        if found is None:
            # Can't determine if entity is new or old
            return 'unk', None
        
        is_mult = found in new_dateproperty_mult2text
        datafield = entity.get(found)
        times = []
        for i in range(len(datafield)):
            try:
                times.append(np.datetime64(correct_time(datafield[i]['value']['time']), 'D'))
            except ValueError:
                continue
        if len(times) == 0:
            return 'unk', None
        datafield = times
        if is_mult:
            reduce = new_dateproperty_mult2text[found]['reduce']
            if reduce == 'oldest':
                value = np.min(datafield)
        else:
            value = datafield[0]

        value_to_put = 'new' if value > old_wikidata_timestamp else 'old'
        return value_to_put, str(value)
        

def keep_only_datavalue(ent_dict:dict):
    if ent_dict is None:
        return
    claims = {}
    for prop_id, statements in ent_dict['claims'].items():
        st = []
        for statement in statements:
            validity = statement.pop('validity')
            qualifiers = statement.pop('qualifiers', None)
            validity['qualifiers'] = qualifiers
            data_value = statement['mainsnak']['datavalue']
            data_value['validity'] = validity
            st.append(data_value)
        claims[prop_id] = st
    ent_dict['claims'] = claims

class DictWithIgnore:
    to_ignore = 'validity'
    labels = ['new', 'old']
    def __init__(self, d : dict, label : str) -> None:
        self.d = d
        self.label = label
        self.ig = d.pop(self.to_ignore, None)
        self.other_ig = None
    
    def __eq__(self, __value: DictWithIgnore) -> bool:
        return self.d == __value.d
    
    def set_other_ig(self, other_ig):
        self.other_ig = other_ig
    
    def restore(self) -> dict:
        self.d[self.to_ignore + '_' + self.label] = self.ig
        if self.other_ig is not None:
            tmp = self.labels.copy()
            tmp.remove(self.label)
            other_label = tmp[0]
            self.d[self.to_ignore + '_' + other_label] = self.other_ig
        return self.d

    def copy(self):
        d = DictWithIgnore(deepcopy(self.d), self.label)
        d.ig = deepcopy(self.ig)
        d.other_ig = deepcopy(self.other_ig)
        return d
    
    def closest_to(self, dicts : list[DictWithIgnore], to_ignore_var : list[str]) -> int:
        def get_value(d, to_ignore_var):
            for x in to_ignore_var:
                d = d[x]
            return d

        equals = [i for i,d in enumerate(dicts) if d == self]
        if len(equals) == 0:
            return -1
        if len(equals) == 1:
            return equals[0]
        equals = [(i,sim_dict(get_value(self.ig, to_ignore_var), get_value(dicts[i].ig, to_ignore_var), depth_penalty=False)) for i in equals]
        closest = max(equals, key=lambda x : x[1])[0]
        return closest



def compute_diff_statements(old_statements : list, new_statements : list) -> list:
    old_statements = [DictWithIgnore(d, 'old') for d in old_statements]
    new_statements = [DictWithIgnore(d, 'new') for d in new_statements]

    

    both, new, old = [],new_statements,[]
    for snak in old_statements:
        try:
            idx = snak.closest_to(new_statements, ['qualifiers'])
            # idx = new_statements.index(snak)
            if idx == -1:
                raise ValueError
            snak.set_other_ig(new_statements[idx].ig)
            new.pop(idx)
            both.append(snak)
            continue
        except ValueError:
            pass
            
        old.append(snak)
    
    both = [d.restore() for d in both]
    new = [d.restore() for d in new]
    old = [d.restore() for d in old]

    diff = []
    for l, flag in [(both, 'both'), (new, 'new'), (old, 'old')]:
        for x in l:
            x['diff_flag'] = flag
            diff.append(x)
    return diff
    
    

def compute_diff(old : dict, new : dict) -> dict:
    diff = {}
    old = {'claims' : {}} if old is None else old
    new = {'claims' : {}} if new is None else new
    for prop_id, old_snaks in old['claims'].items():
        new_snaks = new['claims'].get(prop_id, None) if new is not None else None
        if new_snaks is not None:
            new['claims'].pop(prop_id)
        else:
            new_snaks = []
        diff[prop_id] = compute_diff_statements(old_snaks, new_snaks)
    
    for prop_id, new_snaks in new['claims'].items():
        diff[prop_id] = compute_diff_statements([], new_snaks)
    return diff

old_coll = db['wikidata_old_prep']
new_coll = db['wikidata_new_prep']

diff_coll = db['wikidata_diff']
diff_coll.drop()

visited_new = set()
physically_new_entities = []
coll_physically_new_entities = db['physically_new_entities']
coll_physically_new_entities.drop()

to_push = []
old_coll_size = old_coll.estimated_document_count()
new_coll_size = new_coll.estimated_document_count()

for i, old_ent_dict in enumerate(old_coll.find(), 1):
    if _DEBUG_NUM_ENT > 0 and i > _DEBUG_NUM_ENT:
        break
    diff = {'_id' : old_ent_dict['_id'], 'label' : old_ent_dict.get('label', None)}
    new_ent_dict = new_coll.find_one({'_id' : old_ent_dict['_id']})
    if new_ent_dict is not None:
        visited_new.add(new_ent_dict['_id'])
    keep_only_datavalue(new_ent_dict)
    keep_only_datavalue(old_ent_dict)
    diff['diff'] = compute_diff(old_ent_dict, new_ent_dict)
    
    # Categorize new entities 
    if all(snak['diff_flag'] == 'new' for snaks in diff['diff'].values() for snak in snaks):
        physicality, date_of_creation = new_in_wikidata_categorize(diff['diff'])
        diff['physicality'] = physicality
        if physicality == 'new':
            physically_new_entities.append((diff['_id'], date_of_creation))
        GlobalStat.physicality_count[physicality] += 1
    
    GlobalStat.diff_analysis(diff)
    to_push.append(diff)
    if i % 1000 == 0:
        diff_coll.insert_many(to_push, ordered=False)
        to_push.clear()
        if len(physically_new_entities):
            coll_physically_new_entities.insert_many([{'_id' : x, 'date_of_creation' : y} for x,y in physically_new_entities], ordered=False)
            physically_new_entities.clear()
        print('[Phase 1/2 (Old)] %s/%s (%0.3f%%)' % (i, old_coll_size, i/old_coll_size*100))
        GlobalStat.save(osp.join(STORAGE_FOLDER, 'resources/script_stats/compute_diff.json'))

if len(to_push):
    diff_coll.insert_many(to_push, ordered=False)
    to_push.clear()
if len(physically_new_entities):
    coll_physically_new_entities.insert_many([{'_id' : x} for x in physically_new_entities], ordered=False)
    physically_new_entities.clear()

for i, new_ent_dict in enumerate(new_coll.find(), 1):
    if _DEBUG_NUM_ENT > 0 and i > _DEBUG_NUM_ENT:
        break
    if new_ent_dict['_id'] in visited_new:
        continue
    diff = {'_id' : new_ent_dict['_id'], 'label' : new_ent_dict.get('label', None)}
    keep_only_datavalue(new_ent_dict)
    old_ent_dict = None
    diff['diff'] = compute_diff(old_ent_dict, new_ent_dict)

    # Categorize new entities 
    if all(snak['diff_flag'] == 'new' for snaks in diff['diff'].values() for snak in snaks):
        physicality, date_of_creation = new_in_wikidata_categorize(diff['diff'])
        diff['physicality'] = physicality
        GlobalStat.physicality_count[physicality] += 1
        if physicality == 'new':
            physically_new_entities.append((diff['_id'], date_of_creation))

    
    GlobalStat.diff_analysis(diff)
    to_push.append(diff)
    if i % 1000 == 0:
        diff_coll.insert_many(to_push, ordered=False)
        to_push.clear()
        if len(physically_new_entities):
            coll_physically_new_entities.insert_many([{'_id' : x, 'date_of_creation' : y} for x,y in physically_new_entities], ordered=False)
            physically_new_entities.clear()
        print('[Phase 2/2 (New)] %s/%s (%0.3f%%)' % (i, new_coll_size, i/new_coll_size*100))
        GlobalStat.save(osp.join(STORAGE_FOLDER, 'resources/script_stats/compute_diff.json'))

if len(to_push):
    diff_coll.insert_many(to_push, ordered=False)
    to_push.clear()
if len(physically_new_entities):
    coll_physically_new_entities.insert_many([{'_id' : x, 'date_of_creation' : y} for x,y in physically_new_entities], ordered=False)
    physically_new_entities.clear()
GlobalStat.save(osp.join(STORAGE_FOLDER, 'resources/script_stats/compute_diff.json'))
