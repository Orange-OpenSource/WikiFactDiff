from collections import defaultdict
import multiprocessing as mp
import time
from typing import Any
import indexed_bzip2
import json
from build.config import *
from pymongo import MongoClient
import os
import queue
import argparse
import os.path as osp

cpus = mp.cpu_count() // 1.5
CHUNK_SIZE = 20*2**20
N_CPU_BZ2_DECOMPRESSION = max(1, round(cpus*6/16))
N_CPU_PROCESS_CHUNK = max(1, round(cpus*6/16))
N_CPU_MONGODB_INSERT = max(1, round(cpus*4/16))

_DEBUG_CHUNK_LIMIT = -1

# Properties whose datatypes are in the following list should be ignored:
# - 'url' : URLs
# - 'external-id' ; External IDs
# - 'commonsMedia' : Media that is not text (images, videos, PDFs, documents, etc.)
# - 'globe-coordinate' : Globe coordinates (latitude, longitude)
BANNED_DATATYPES = ('url', 'external-id', 'commonsMedia', 'globe-coordinate')

desc ="""
This script will:
1. Uncompress the given Wikidata JSON dump file
2. Remove unecessary properties from entities such as (media, url and external ids properties) and keep only truthy statements "https://www.mediawiki.org/wiki/Wikibase/Indexing/RDF_Dump_Format#Truthy_statements"
3. Push the result to a MongoDB dataset

MongoDB credentials can be found in config.py
"""
parser = argparse.ArgumentParser(description=desc)

# parser.add_argument("--wikidata_json_path", help='Path to the JSON wikidata dump to process', type=str, required=True)
# parser.add_argument('--coll_name', help="Collection name in MongoDB where the JSON will be pushed", type=str, required=True)
parser.add_argument("--version", help='Wikidata version to process', type=str, required=True, choices=['old', 'new'])
parser.add_argument("--n_cpu_bz2_decompression", help='Number of parallel processes that will be used for BZ2 decompression. Defaults to {}'.format(N_CPU_BZ2_DECOMPRESSION), type=int, default=N_CPU_BZ2_DECOMPRESSION)
parser.add_argument("--n_cpu_process_chunk", help='Number of parallel processes that will be used to process chunks. Defaults to {}'.format(N_CPU_PROCESS_CHUNK), type=int, default=N_CPU_PROCESS_CHUNK)
parser.add_argument("--n_cpu_mongodb_insert", help='Number of parallel processes that will be used to push data to MongoDB. Defaults to {}'.format(N_CPU_MONGODB_INSERT), type=int, default=N_CPU_MONGODB_INSERT)
parser.add_argument("--chunk_size", help='Size of uncompressed chunks in bytes. Defaults to {} which means {}MB'.format(CHUNK_SIZE, round(CHUNK_SIZE/2**20,4)), type=int, default=CHUNK_SIZE)
parser.add_argument("--truthy", help='If specified, get the truthy version of Wikidata, i.e., keep only statements with preferred rank when provided and remove deprecated statements. If unspecified, remove only deprecated statements',action='store_true')
parser.add_argument("--stat_only", help='Do not create any collection and do not push data to MongoDB. The only output of this script is the stats computed from the given Wikidata dump. If activated, there will be a substantial gain execution time',action='store_true')

def process_object(obj : dict, truthy : bool, stats : dict):
    if 'type' not in obj:
        return False
    if obj['type'] == 'item':
        return process_entity(obj, truthy, stats)
    elif obj['type'] == 'property':
        return process_property(obj, stats)
    return False

def extract_entities(statements : list[dict]):
    res = []
    for statement in statements:
        mainsnak = statement['mainsnak']
        if mainsnak['snaktype'] != 'value':
            res.append(mainsnak['snaktype'])
            continue
        try:
            res.append(mainsnak['datavalue']['value']['id'])
        except KeyError as e:
            print('Key missing : %s' % e.args[0])
    return res

def process_property(entity : dict, stats : dict):
    try:
        entity['_id'] = entity['id']
        del entity['id']
    except KeyError:
        return False
    try:
        _tmp = entity['labels']
        if 'en' in _tmp:
            entity['label'] = _tmp['en']['value']
        else:
            k = next(iter(_tmp.keys()))
            entity['label'] = _tmp[k]['value']
        entity['label_upper'] = entity['label'].upper()
    except (KeyError, StopIteration):
        pass
    if 'labels' in entity: del entity['labels']

    try:
        _tmp = entity['descriptions']
        if 'en' in _tmp:
            entity['description'] = _tmp['en']['value']
        else:
            k = next(iter(_tmp.keys()))
            entity['description'] = _tmp[k]['value']
    except (KeyError, StopIteration):
        pass
    if 'descriptions' in entity: del entity['descriptions']

    try:
        entity['aliases'] = tuple(x['value'] for x in entity['aliases']['en'])
    except KeyError:
        entity['aliases'] = []

    try:
        entity['enwiki_title'] = entity['sitelinks']['enwiki']['title']
        stats['n_wiki_entities'] = stats.get('n_wiki_entities', 0) + 1
    except KeyError:
        pass
    if 'sitelinks' in entity : del entity['sitelinks']

    for i in ('pageid', 'ns', 'title'):
        if i in entity:
            del entity[i]
    stats['n_entities'] = stats.get('n_entities', 0) + 1
    if 'P31' not in entity['claims']:
        stats['type_count']['none'] += 1
    for k, v in entity['claims'].items():
        stats['relation_count'][k] += 1
        if k == 'P31':
            for t in extract_entities(v):
                stats['type_count'][t] += 1
        stats['n_triples'] = stats.get('n_triples', 0) + len(v)
        stats['n_groups'] = stats.get('n_groups', 0) + 1
        for d in v:
            stats['datatype_n_%s' % d['mainsnak']['datatype']] = stats.get('datatype_n_%s' % d['mainsnak']['datatype'], 0) + 1
    return True

def process_entity(entity : dict, truthy : bool, stats : dict):
    try:
        entity['_id'] = entity['id']
        del entity['id']
    except KeyError:
        return False
    try:
        _tmp = entity['labels']
        if 'en' in _tmp:
            entity['label'] = _tmp['en']['value']
        else:
            k = next(iter(_tmp.keys()))
            entity['label'] = _tmp[k]['value']
        entity['label_upper'] = entity['label'].upper()
    except (KeyError, StopIteration):
        pass
    if 'labels' in entity: del entity['labels']

    try:
        _tmp = entity['descriptions']
        if 'en' in _tmp:
            entity['description'] = _tmp['en']['value']
        else:
            k = next(iter(_tmp.keys()))
            entity['description'] = _tmp[k]['value']
    except (KeyError, StopIteration):
        pass
    if 'descriptions' in entity: del entity['descriptions']

    try:
        entity['aliases'] = tuple(x['value'] for x in entity['aliases']['en'])
    except KeyError:
        pass

    try:
        entity['enwiki_title'] = entity['sitelinks']['enwiki']['title']
        stats['n_wiki_entities'] = stats.get('n_wiki_entities', 0) + 1
    except KeyError:
        pass
    if 'sitelinks' in entity : del entity['sitelinks']

    for i in ('pageid', 'ns', 'title'):
        if i in entity:
            del entity[i]

    statements_to_keep = []
    stats['n_entities'] = stats.get('n_entities', 0) + 1
    if 'P31' not in entity['claims']:
        stats['type_count']['none'] += 1
    for k, v in entity['claims'].items():
        stats['relation_count'][k] += 1
        if k == 'P31':
            for t in extract_entities(v):
                stats['type_count'][t] += 1
        stats['n_triples'] = stats.get('n_triples', 0) + len(v)
        stats['n_groups'] = stats.get('n_groups', 0) + 1
        to_keep = []
        ranks = []
        for i,d in enumerate(v):
            stats['datatype_n_%s' % d['mainsnak']['datatype']] = stats.get('datatype_n_%s' % d['mainsnak']['datatype'], 0) + 1
            try:
                if d['mainsnak']['datatype'] in BANNED_DATATYPES:
                    stats['n_triples_banned_types'] = stats.get('n_triples_banned_types', 0) + 1
                    continue
                elif d['rank'] == 'deprecated':
                    stats['n_triples_deprecated'] = stats.get('n_triples_deprecated', 0) + 1
                    continue
                to_keep.append(i)
                ranks.append(d['rank'])
                del d['references']
                del d['id']
            except (NameError, KeyError):
                pass
            
        if truthy:
            if len(v) and any(x == 'preferred' for x in ranks):
                to_keep = [to_keep[j] for j in range(len(to_keep)) if ranks[j] == 'preferred']
        if len(to_keep):
            # Filter properties in qualifiers (using constant BANNED_DATATYPES)
            # EDIT (8 May 2023) : Do not filter banned properties because we need them to know if an RDF is restricted by its qualifiers
            for i in to_keep:
                to_filter = v[i]
                if 'qualifiers' in to_filter:
                    # to_filter['qualifiers'] = {k:[x for x in v if x['datatype'] not in BANNED_DATATYPES] for k,v in to_filter['qualifiers'].items()}
                    to_filter['qualifiers'] = {k:[x for x in v] for k,v in to_filter['qualifiers'].items()}

                    # Remove empty qualifiers
                    to_filter['qualifiers'] = {k : v for k,v in to_filter['qualifiers'].items() if len(v)}

                    if len(to_filter['qualifiers']) == 0:
                        del to_filter['qualifiers']
                        
            statements_to_keep.append((k, to_keep))
    _tmp = entity['claims']
    entity['claims'] = {k:[_tmp[k][i] for i in to_keep] for k, to_keep in statements_to_keep}
    stats['n_groups_post'] = stats.get('n_groups_post', 0) + len(statements_to_keep)
    return True

def process_io(queue_io, path, file_position : float, n_cpu : int, n_cpu_process : int, chunk_size : int):
    set_proc_name('WFD:IO')
    skip_list = b'\n,]'
    file = indexed_bzip2.open( path, parallelization = n_cpu)
    file.read(2)
    count_chunks = 0
    while True:
        if _DEBUG_CHUNK_LIMIT > 0 and count_chunks >= _DEBUG_CHUNK_LIMIT:
            break
        chunk = file.read(chunk_size) + file.readline()
        if len(chunk) == 0:
            break
        
        # Remove useless data
        i = 0
        try:
            while chunk[i-1] in skip_list:
                i -= 1
        except IndexError:
            pass
        if i != 0:
            chunk = chunk[:i]
        if len(chunk):
            queue_io.put(chunk)
            pos = file.tell_compressed()
            if pos != 0:
                file_position.value = pos / 8 / 2**30
            else:
                file_position.value = max(file.available_block_offsets()) / 8 / 2**30
        count_chunks += 1
    for _ in range(n_cpu_process):
        queue_io.put(None)
    file.close()
    

def process_stats(queue_io : mp.Queue, queue_mongo : mp.Queue, file_position : float, file_size : float):
    set_proc_name('WFD:STATS')
    file_size = round(file_size / 2**30, 4)
    while True:
        # os.system('cls' if os.name == 'nt' else 'clear')
        print()
        print('IO Queue size =', queue_io.qsize())
        print('Mongo Queue size =', queue_mongo.qsize())
        print('Progress : {}GB / {}GB'.format(round(file_position.value, 4), file_size))
        time.sleep(1)

def process_chunk(queue_io : mp.Queue, queue_mongo : mp.Queue, n_process_alive : int, truthy : bool, stats_queue : mp.Queue, stat_only : bool):
    set_proc_name('WFD:CHUNK')
    stats = {'relation_count' : defaultdict(lambda : 0), 'type_count' : defaultdict(lambda : 0)}
    while True:
        chunk = queue_io.get()
        if chunk is None:
            break
        data = [json.loads(x.rstrip(b',')) for x in chunk.split(b'\n')]
        data = [obj for obj in data if process_object(obj, truthy, stats)]
        if not stat_only:
            queue_mongo.put(data)
    stats['relation_count'] = dict(stats['relation_count'])
    stats['type_count'] = dict(stats['type_count'])
    stats_queue.put(stats)
    n_process_alive.value -= 1
    

def process_mongo(queue_mongo : mp.Queue, n_process_chunk_alive : int, coll_name : str):
    set_proc_name('WFD:MONGO')
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    collection = db[coll_name]
    
    while True:
        if n_process_chunk_alive.value == 0 and queue_mongo.empty():
            break
        try:
            to_insert = queue_mongo.get(timeout=1)
        except queue.Empty:
            continue

        collection.insert_many(to_insert, ordered=False)

def set_proc_name(newname):
    try:
        from ctypes import cdll, byref, create_string_buffer
        libc = cdll.LoadLibrary('libc.so.6')
        buff = create_string_buffer(len(newname)+1)
        buff.value = newname
        libc.prctl(15, byref(buff), 0, 0, 0)
    except:
        pass

    
if __name__ == '__main__':
    t1 = time.time()
    args = parser.parse_args()
    mp.set_start_method('spawn')
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    version = args.version
    coll_name = "wikidata_%s_json" % version
    collection = db[coll_name]
    if not args.stat_only:
        collection.drop()
        collection.create_index([('label_upper', 1)])

    io_q = mp.Queue(args.n_cpu_process_chunk)
    queue_mongo = mp.Queue(args.n_cpu_process_chunk)
    n_process_chunk_alive = mp.Value('i', args.n_cpu_process_chunk)
    truthy = args.truthy
    file_position = mp.Value('d', 0)
    path = osp.join(STORAGE_FOLDER, '%s_wikidata.json.bz2' % version)

    stats_queue = mp.Queue()

    p_io = mp.Process(target=process_io, args=(io_q, path, file_position, args.n_cpu_bz2_decompression, args.n_cpu_process_chunk, args.chunk_size))
    p_c = [mp.Process(target=process_chunk, args=(io_q,queue_mongo,n_process_chunk_alive, truthy, stats_queue, args.stat_only)) for _ in range(args.n_cpu_process_chunk)]
    p_stats = mp.Process(target=process_stats, args=(io_q,queue_mongo, file_position, os.path.getsize(path)))
    if not args.stat_only:
        p_mongo = [mp.Process(target=process_mongo, args=(queue_mongo,n_process_chunk_alive, coll_name)) for _ in range(args.n_cpu_mongodb_insert)]

    p_stats.start()
    p_io.start()
    [x.start() for x in p_c]
    if not args.stat_only:
        [x.start() for x in p_mongo]

    p_io.join()
    stats = [stats_queue.get() for _ in range(args.n_cpu_process_chunk)]
    [x.join() for x in p_c]
    if not args.stat_only:
        [x.join() for x in p_mongo]
    p_stats.kill()

    relation_count = defaultdict(lambda : 0)
    type_count = defaultdict(lambda : 0)
    stats_dd = defaultdict(lambda : 0)
    for stat in stats:
        for k, v in stat.items():
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    if k == 'relation_count':
                        relation_count[k2] += v2
                    else:
                        type_count[k2] += v2
            else:
                stats_dd[k] += v
    stats_dd['relation_count'] = relation_count
    stats_dd['type_count'] = type_count
    os.makedirs(osp.join(STORAGE_FOLDER, 'resources/script_stats'), exist_ok=True)
    with open(osp.join(STORAGE_FOLDER, 'resources/script_stats/process_json_dump_%s.json' % coll_name), 'w') as f:
        json.dump(stats_dd, f, indent=4, sort_keys=True)


    t2 = time.time()
    print('\n\nFinished !')
    print('Time elapsed =', round(t2-t1, 2), 'sec')
