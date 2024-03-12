from collections import Counter, defaultdict
from build.utils.wd import db, get_info_wikidata, get_value, get_single_valued_properties
from build.gpt3_5_verbalization.verbalize_wikidata import PropertyInfoRetriever, Entity, Property, Literal, KnowledgeTriple
from typing import Union
from build.utils.ku import Feature, classify_algorithm_lite
from build.verbalize_wikifactdiff.utils import Verbalizer
from multiprocessing.pool import ThreadPool
from threading import Lock
import random
import json
import os.path as osp

from build.config import STORAGE_FOLDER

SAVE_PATH = osp.join(STORAGE_FOLDER, 'wikifactdiff_dirty.jsonl')

def value_to_object(value : dict) -> Union[Literal, Entity]:
    obj, t = get_value(value, get_info_dict_for_entities=True, add_type=True, add_unit=True, version='new')
    if obj is None or len(obj) == 0: 
        obj, t = get_value(value, get_info_dict_for_entities=True, add_type=True, add_unit=True, version='old')
        if obj is None or len(obj) == 0:
            return None
    if isinstance(obj, dict):
        obj = Entity(obj["id"], obj['name'], obj['description'])
    else:
        obj = Literal(obj, t)
    return obj

class Finish(Exception):
    pass

def get_nearest_triples(knn, ent_id : str, rel : Property, k = 500, banned_ent : list[str] = []) -> list[KnowledgeTriple]:
    # Get the 10 most similar entities to ent_id that have relation rel and collect the group (neighbor_ent, rel) and put all triples a list and return it
    vector = knn.get_vectors(ent_id)
    if vector is None:
        return [], []
    
    neighbor_entities, distances = knn.get_nearest(vector,k=k)
    to_remove = [i for i in range(len(neighbor_entities)) if neighbor_entities[i] == ent_id or neighbor_entities[i] in banned_ent]
    neighbor_entities = [x for i,x in enumerate(neighbor_entities) if i not in to_remove]
    distances = [x for i,x in enumerate(distances) if i not in to_remove]

    step = 20
    result = []
    dists = []
    n_neighobrs_found = 0
    try:
        for i in range(0, len(neighbor_entities), step):
            ents = neighbor_entities[i:i+step]
            ds = distances[i:i+step]
            ent_dicts = [x for x in db['wikidata_old_prep'].find({'_id': {'$in' : ents}, 'claims.%s' % rel.id : {'$exists' : 1}}, {'claims.%s' % rel.id : 1, "label" : 1, "description" : 1})]

            # Remove entities with no labels
            ent_dicts = [x for x in ent_dicts if x.get('label', None) is not None]

            ents_found = set(x['_id'] for x in ent_dicts)
            ds_found = []
            for i in range(len(ds)):
                if ents[i] in ents_found:
                    ds_found.append(ds[i])
            for ent, d in zip(ent_dicts, ds_found):
                prop_values = ent['claims'].get(rel.id, None)
                n_neighobrs_found += 1
                for snak in prop_values:
                    feature = Feature(snak)
                    if classify_algorithm_lite(feature, version='old') != 'keep':
                        continue
                    sub = Entity(ent['_id'], ent.get('label', None), ent.get('description', None))
                    obj = value_to_object(snak['mainsnak']['datavalue']['value'])
                    if obj is None:
                        n_neighobrs_found -= 1
                        break
                    triple = KnowledgeTriple(sub, rel, obj)
                    result.append(triple)
                    dists.append(float(d))
                if n_neighobrs_found >= 10:
                    raise Finish
    except Finish:
        pass
    return result, dists

def _is_replace(objects : list[dict]):
    c = Counter([x['decision'] for x in objects])
    return len(c) == 2 and c['learn'] == 1 and c['forget'] == 1

def verbalize_and_compact(verbalizer : Verbalizer, triples_decisions : list[tuple[KnowledgeTriple, str]], neighbor_triples : list[KnowledgeTriple], distances: list[float], ent_is_ph_new : bool, rel_is_functional : bool, ent_imp : float) -> dict:
    triples, decisions = list(zip(*triples_decisions))
    triple = triples[0]

    verbs = verbalizer.verbalize(triple, exclude=['object'])
    if verbs is None:
        return None

    objects_decisions = []
    for t, d in zip(triples, decisions):
        dd = t.object.to_dict()
        dd['decision'] = d
        objects_decisions.append(dd)
    
    neighbors = defaultdict(list)
    for t, dist in zip(neighbor_triples, distances):
        verbs_t = verbalizer.verbalize(t, exclude=['object'])
        neighbors[t.subject].append ({
            'object' : t.object.to_dict(),
            'prompt' : random.choice(verbs_t),
            'dist' : dist
        })
    neighbors = [{'objects' : v, 'subject' : k.to_dict()} for k,v in neighbors.items()]
    

    instance = {
        'subject' : triple.subject.to_dict(),
        'relation' : triple.relation.to_dict(),
        'update_prompt' : verbs[0],
        'generalization_prompts' : verbs[1:],
        'subject_is_ph_new' : ent_is_ph_new,
        'relation_is_temp_func' : rel_is_functional,
        'is_replace' : _is_replace(objects_decisions),
        'subject_popularity' : ent_imp,
        'objects' : objects_decisions,
        'neighborhood' : neighbors,
    }
    return instance


def process_one_group(x):
    ent_id = x['_id']['ent_id']
    ent_info = get_info_wikidata(ent_id, version='new')
    prop_id = x['_id']['prop_id']
    with lock:
        prop_info = prop_info_ret.retrieve(prop_id)
    sub = Entity(ent_info['id'], ent_info['name'], ent_info['description'])
    rel = Property(prop_info['id'], prop_info['name'], prop_info['description'])
    ent_ph_new = ent_id in ph_new_entities
    rel_is_functional = prop_id in functional_relations
    
    snaks = [y for y in x['snaks']]
    triples_decisions = []
    banned_ent = []
    for snak in snaks:
        value = snak['value']
        try:
            obj = value_to_object(value)
            if obj is None:
                break
            triple = KnowledgeTriple(sub, rel, obj)
            triples_decisions.append((triple, snak['decision']))
            if snak['decision'] != 'keep':
                if isinstance(obj, Entity):
                    banned_ent.append(obj.id)
        except:
            print('Serious error:', ent_info, prop_id)
            break
    if len(triples_decisions) != len(snaks):
        return None
    
    neighbor_triples, distances = get_nearest_triples(knn, ent_id, rel, banned_ent=banned_ent)
    if neighbor_triples is None:
        return None

    instance = verbalize_and_compact(verbalizer, triples_decisions, neighbor_triples, distances, ent_ph_new, rel_is_functional, x['ent_imp'])
    return instance

def main():
    # Retrieve progress
    seen_groups = set()
    print('Retrieving already verbalized groups...')
    try:
        f = open(SAVE_PATH, 'r')
        instances = [json.loads(x) for x in f.readlines()]
        seen_groups.update((x['subject']['id'], x['relation']['id']) for x in instances)
        f.close()
        print('%s groups retrieved.' % len(seen_groups))
    except FileNotFoundError:
        print("File not found! Starting from scratch.")
        pass
    
    f = open(SAVE_PATH, 'a')

    cursor = db['wkd_fd'].find()
    n = db['wkd_fd'].estimated_document_count()
    knn.load_index()
    instances = []
    thread_pool = ThreadPool(8)
    for i,instance in enumerate(cursor, 1):
        ent_id, prop_id = instance['_id']['ent_id'], instance['_id']['prop_id']
        # Debug
        # if ent_id != 'Q615' or prop_id != 'P286':
        #     continue
        if (ent_id, prop_id) in seen_groups:
            continue
        instance = process_one_group(instance)
        if instance is None:
            continue
        # instances.append(instance)
        f.write(json.dumps(instance) + '\n')
        f.flush()
        if i % 50 == 0:
            print('Progress : %s/%s groups (%0.2f%%)' % (i, n, i/n*100))
    f.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_method', type=str, help='What Approximate nearest neighbor method to use? Can take two values : "sparse", "dense".', required=True, choices=['sparse', 'dense'])
    # parser.add_argument('option', type=str, help='What type of dataset you want to create? (For which experiment) Can take three values : "ku","nn_eval"', required=True)
    args = parser.parse_args()

    count_entities = 0
    ph_new_entities = set(x['_id'] for x in db['physically_new_entities'].find())
    functional_relations = set(get_single_valued_properties())
    if args.ann_method == 'dense':
        from build.find_neighbors.knn_dense import KNearestNeighbors
    else:
        from build.find_neighbors.knn_sparse_nmslib import KNearestNeighbors
    knn = KNearestNeighbors()
    verbalizer = Verbalizer()
    lock = Lock()
    prop_info_ret = PropertyInfoRetriever(versions=['new', 'old'])


    main()