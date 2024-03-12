# Compute the importance of entities and properties
# Importance(entity) = log(median of last 12 months consultations + 1)
# Importance(property in an entity) = log(max(number of entities of the same 'instance of's that have this property) + 1) * importance(entity)
# Importance(entity - consultation not found) = the median of the importance of same 'instance of'
# For the last 12 months consultations, start counting from the first non zero month to not penalize new entities

import argparse
import queue
from pymongo import MongoClient
from build.config import *
import numpy as np
import bson
from collections import defaultdict
import multiprocessing as mp
import os
import time
import json
import os.path as osp

COLLECTION_NAME = 'wikidata_diff'
N_DOCUMENT_CHUNK = 500
N_PROCESS_CHUNK = max(1, round(mp.cpu_count() * 3/4))
# N_PROCESS_CHUNK = 1
_N_DEBUG = -1

def set_proc_name(newname):
    # This function sets the process name in the OS (use 'top' command in Linux to see results)
    from ctypes import cdll, byref, create_string_buffer
    libc = cdll.LoadLibrary('libc.so.6')
    buff = create_string_buffer(len(newname)+1)
    buff.value = newname
    libc.prctl(15, byref(buff), 0, 0, 0)

def default_to_regular(d):
    """Nested defaultdict to regular dict

    Args:
        d (defaultdict)

    Returns:
        dict
    """
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

def month_consults2importance(month_consults : list[dict], list_yearmonths : list[str]) -> float:
    month_consults = {x['year_month'] : x['n'] for x in month_consults}
    for yearmonth in list_yearmonths:
      if yearmonth not in month_consults:
          month_consults[yearmonth] = 0
    keys_sorted = list(month_consults.keys())
    keys_sorted.sort()
    i = 0
    for key in keys_sorted:
        if month_consults[key] != 0:
            break
        i += 1
    keys_sorted = keys_sorted[i:]
    consults = [month_consults[key] for key in keys_sorted]
    imp = np.median(consults)
    return imp

def process_io(vars):
    set_proc_name(b'Process IO')
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    for i, chunk in enumerate(db[COLLECTION_NAME].find_raw_batches(batch_size=N_DOCUMENT_CHUNK)):
        if _N_DEBUG > 0 and i > _N_DEBUG:
            break
        vars['queue'].put(chunk)
        vars['position'].value += N_DOCUMENT_CHUNK
    vars['alive'].value = False

def process_chunk(vars):
    set_proc_name(b'Process Chunk')
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    not_found_entities = []
    entities_importance = []
    entities_types_importance_stats = defaultdict(lambda : {'sum' : 0, 'n' : 0})
    entity_type_property_importance_stats = defaultdict(lambda : defaultdict(lambda : 0))
    entities_types = {}

    version, anti_version = vars['version'], vars['anti_version']
    list_yearmonths = vars['list_yearmonths']

    while vars['io_alive'].value or vars['queue_io'].qsize():
        try:
            chunk = vars['queue_io'].get(timeout=1)
        except queue.Empty:
            continue
        chunk_full = bson.decode_all(chunk)
        chunk = [x['_id'] for x in chunk_full]
        chunk_full = {x['_id'] :x for x in chunk_full}
        # enwiki_title contains the title of the wikipedia at the timestamp NEW so it could not exist in the OLD timestamp, that's why we add the condition "enwiki_title" : {'$exists' : 1}
        chunk = list(db['wikidata_%s_json' % version].find({'_id' : {'$in' : chunk}, "enwiki_title" : {'$exists' : 1}}, {'enwiki_title' : 1}, batch_size=N_DOCUMENT_CHUNK))
        chunk_full = {x['_id'] : chunk_full[x['_id']] for x in chunk}
        pointers = {x['enwiki_title'] : x for x in chunk}
        set_entities = set(chunk_full.keys())
        entities_importance_ = []
        ent2types = {}
        consults = list(db['%s_wikipedia_month_consultation' % version].find({'_id' : {'$in' : list(pointers.keys())}}, {'_id' : 1, 'month_consults' : 1}, batch_size=N_DOCUMENT_CHUNK))
        for d in consults:
            ent = pointers[d['_id']]
            importance = month_consults2importance(d['month_consults'], list_yearmonths)
            ent['importance'] = importance
            del ent['enwiki_title']
            set_entities.remove(ent['_id'])
            entities_importance_.append(ent)
            
            ent = chunk_full[ent['_id']]
            diff = ent['diff']
            # Entity type importance stats
            ent_types = []
            if 'P31' not in diff:
                s = entities_types_importance_stats[None]
                s['sum'] += importance
                s['n'] += 1
            else:
                for statement in [s for s in diff['P31'] if s['diff_flag'] != anti_version]:
                    id_ = statement['value']['id']
                    s = entities_types_importance_stats[id_]
                    s['sum'] += importance
                    s['n'] += 1
                    ent_types.append(id_)
            ent2types[ent['_id']] = ent_types

        for ent in chunk_full.values():
            # Entity-type property importance stats
            diff = ent['diff']
            ent_types = ent2types.get(ent['_id'])
            if ent_types is None:
                ent_types = []
                if 'P31' in diff:
                    for statement in [s for s in diff['P31'] if s['diff_flag'] != anti_version]:
                        id_ = statement['value']['id']
                        ent_types.append(id_)
                ent2types[ent['_id']] = ent_types
            
            for prop, statements in diff.items():
                for statement in [s for s in statements if s['diff_flag'] != anti_version]:
                    if len(ent_types):
                        for t in ent_types:
                            entity_type_property_importance_stats[t][prop] += 1
                    else:
                        entity_type_property_importance_stats[None][prop] += 1
                    break

        not_found_entities.extend(set_entities)
        entities_importance.extend(entities_importance_)
        entities_types.update(ent2types)
    vars['queue_info'].put({'not_found_entities' : not_found_entities, # Entities that were not found using month consult database
                            'entities_importance' : entities_importance,  # Entities that were found and their corresponding importance
                            'entities_types_importance_stats' : default_to_regular(entities_types_importance_stats), # Importance of each entity type
                            'entity_type_property_importance_stats' : default_to_regular(entity_type_property_importance_stats), # Importance of properties in entity types
                            'entities_types' : entities_types # Types of each entity
                            })
    vars['n_alive'].value -= 1
    
def process_stats(phase_name : str, position : int, n_documents : int, queues_dict : dict[str, mp.Queue]):
    """A process that prints progress and queues sizes every second

    Args:
        position (int): Shared integer variable. It represents how many wikipedia entities were processed 
        queues_dict (dict[str, mp.Queue]): A dictionary of queues (Key = Queue name, Value = Queue object)
    """
    set_proc_name(b'Process Stats')
    start_time = time.time()
    while True:
        # os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print('Phase :', phase_name)
        print('Progress : {}/{} wikipedia entities processed'.format(position.value, n_documents))
        print('Speed : {} entities/sec'.format(round(position.value/(time.time() - start_time), 2)))
        for name, q in queues_dict.items():
            print(f'{name} size = {q.qsize()}')
        time.sleep(2)


def main(version : str):
    anti_version = 'old' if version == 'new' else 'new'

    WIKIDATA_DATE = OLD_WIKIDATA_DATE if version == 'old' else NEW_WIKIDATA_DATE
    # Computing yearmonths
    list_yearmonths = []
    one_month = np.timedelta64(1, 'M')
    new_date = np.datetime64(
        WIKIDATA_DATE[:4] + '-' + WIKIDATA_DATE[4:6])
    for _ in range(REWIND_N_MONTHS):
        new_date = new_date - one_month
        year = new_date.astype('datetime64[Y]').astype(int) + 1970
        month = str(new_date.astype('datetime64[M]').astype(int) % 12 + 1)
        month = '0' + month if len(month) < 2 else month
        list_yearmonths.append('%s%s' % (year, month))


    mp.set_start_method('spawn')
    t1 = time.time()
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    n_documents = db[COLLECTION_NAME].count_documents({})


    ################## PHASE 1 ################## 
    process_io_vars = {
        'alive' : mp.Value('b',True),
        'queue' : mp.Queue(N_PROCESS_CHUNK),
        'position' : mp.Value('i', 0)
    }
    p_io = mp.Process(target=process_io, args=(process_io_vars,))

    process_chunk_vars = {
        'n_alive' : mp.Value('i', N_PROCESS_CHUNK),
        'queue_io' : process_io_vars['queue'],
        'queue_info' : mp.Queue(N_PROCESS_CHUNK),
        'io_alive' : process_io_vars['alive'],
        'version' : version,
        'anti_version' : anti_version,
        'list_yearmonths' : list_yearmonths
    }

    ps_chunk = [mp.Process(target=process_chunk, args=(process_chunk_vars,)) for _ in range(N_PROCESS_CHUNK)]
    
    queue_dict = {
        'Queue IO' : process_io_vars['queue'],
        'Queue Info' : process_chunk_vars['queue_info']
    }
    p_stat = mp.Process(target=process_stats, args=('Wikidata %s - Acquiring importance stats (entity types + property)' % version.upper(), process_io_vars['position'], n_documents, queue_dict))
    print('Spawning the processes... (If it takes some time, it\'s normal)')
    p_io.start()
    [p.start() for p in ps_chunk]
    p_stat.start()
    p_io.join()

    # Join info from different processes
    entities_types_importance_stats = defaultdict(lambda : {'sum' : 0, 'n' : 0})
    entity_type_property_importance_stats = defaultdict(lambda : defaultdict(lambda : 0))
    not_found_entities = []
    entities_importance = []
    entities_types = {}
    for _ in range(N_PROCESS_CHUNK):
        info = process_chunk_vars['queue_info'].get()
        not_found_entities.extend(info['not_found_entities'])
        entities_importance.extend(info['entities_importance'])
        for k, v in info['entities_types_importance_stats'].items():
            for k2, v2 in v.items():
                entities_types_importance_stats[k][k2] += v2
        for k, v in info['entity_type_property_importance_stats'].items():
            for k2, v2 in v.items():
                entity_type_property_importance_stats[k][k2] += v2
        entities_types.update(info['entities_types'])
    entities_types_importance_stats = {k:np.log(v['sum'] / v['n'] + 1) for k, v in entities_types_importance_stats.items()}
    for k, v in entity_type_property_importance_stats.items():
        for k2, v2 in v.items():
            entity_type_property_importance_stats[k][k2] = np.log(v2+1)
    for i in range(len(entities_importance)):
        entities_importance[i]['importance'] = np.log(entities_importance[i]['importance']+1)
    
    # Save (entity type, property) importance in MongoDB
    print('\n\n')
    print('Push Entity-type-property importance...')
    ent_prop_imp = [{'_id' : {'ent_type' : k, 'prop' : k2}, 'imp' : v2 } for k, v in entity_type_property_importance_stats.items() for k2, v2 in v.items()]
    db['%s_entity_type_property_importance' % version].drop()
    db['%s_entity_type_property_importance' % version].insert_many(ent_prop_imp, ordered=False)
    del ent_prop_imp

    # Saving stats
    stats = {'not_found_entities' : len(not_found_entities), 'n_entities' : n_documents, 'found_entities' : len(entities_importance)}
    # assert stats['not_found_entities'] + stats['found_entities'] == stats['n_entities']
    with open(osp.join(STORAGE_FOLDER, 'resources/script_stats', osp.splitext(osp.basename(__file__))[0] + '_stats.json'), 'w') as f:
        json.dump(stats, f)

    
    [p.join() for p in ps_chunk]
    p_stat.kill()
    
    ################## PHASE 2 ##################
    print('Computing entity importance for not-found entities (this can take sometime)...')
    entities_importance = {x['_id'] : x['importance'] for x in entities_importance}
    med_entities_importance = np.median(list(entities_importance.values()))
    def get_ent_imp(ent_id):
        try:
            types = entities_types[ent_id]
            if len(types):
                types = [t for t in types if t in entities_types_importance_stats]
                if len(types) == 0:
                    # If a entity has types but are notfound in the importance types dictionary, assign to them overall median of entity importance
                    return {'ent_imp' : med_entities_importance, 'types' : types, 'imp_found_using' : 'overall_median_imp'}
        except KeyError:
            types = [None]
        if len(types) == 0:
            types = [None]
        if ent_id not in entities_importance:
            if types == [None]:
                ent_imp = entities_types_importance_stats[None]
                imp_found_using = "ent_type_imp_dictionary_none_type"
            else:
                ent_imp = max(entities_types_importance_stats[t] for t in types)
                imp_found_using = "ent_type_imp_dictionary_max_type"
            return {'ent_imp' : ent_imp, 'types' : types, 'imp_found_using' : imp_found_using}
        return {'ent_imp' : entities_importance[ent_id], 'types' : types, 'imp_found_using' : 'ent_imp_dictionary'}

    entities_importance_final = []
    for ent_id in not_found_entities:
        assert ent_id not in entities_importance
        imp = get_ent_imp(ent_id)
        imp['_id'] = ent_id
        entities_importance_final.append(imp)
    
    for ent_id, imp_ in entities_importance.items():
        imp = get_ent_imp(ent_id)
        assert imp['ent_imp'] == imp_
        imp['_id'] = ent_id
        entities_importance_final.append(imp)


    print('Push entities type importance...')
    entities_types_importance_stats = [{'_id' : k, 'importance' : v} for k,v in entities_types_importance_stats.items()]
    db['%s_entities_types_importance' % version].drop()
    db['%s_entities_types_importance' % version].insert_many(entities_types_importance_stats, ordered=False)

    print('Push entities type...')
    db['%s_entities_types' % version].drop()
    entities_types = [{'_id' : k, 'types' : v} for k,v in entities_types.items()]
    db['%s_entities_types' % version].insert_many(entities_types)

    print('Push entities importance (this can take sometime)...')
    db['%s_entities_importance' % version].drop()
    db['%s_entities_importance' % version].create_index([('ent_imp', -1)])
    db['%s_entities_importance' % version].insert_many(entities_importance_final)

    t2 = time.time()
    print('Processing time = %f sec' % round(t2-t1, 1))
    print('Finished !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["old", 'new'],
        help="Timestamp at which the importance is computed. Must be in ['old', 'new']",
        required=True
    )
    args = parser.parse_args()
    t1 = time.time()
    main(version=args.version)
    print('Execution time = %ssec' % (time.time() - t1))
