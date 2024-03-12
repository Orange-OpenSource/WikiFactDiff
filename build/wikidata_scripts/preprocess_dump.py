import argparse
import os
import time

from frozendict import frozendict
from build.config import STORAGE_FOLDER
from build.utils.wd import db, get_single_valued_properties, get_info_wikidata, get_direct_indirect_types, get_value
from build.utils.ku import get_start_end_time, get_point_in_time, old_wikidata_timestamp, new_wikidata_timestamp
from collections import defaultdict
import json
import os.path as osp

# Script tested by hand on Kanye West entity and on France entity (for 'population' property)

_DEBUG_NUM_ENT = -1

property_object_counts = defaultdict(lambda : 0)

class GlobalStat:
    type_count = defaultdict(lambda : 0)
    relation_count = defaultdict(lambda : 0)
    relation_count_all = defaultdict(lambda : 0)
    post_type_count = defaultdict(lambda : 0)
    post_relation_count = defaultdict(lambda : 0)
    post_relation_count_all = defaultdict(lambda : 0)
    n_wiki_entities = 0
    post_n_wiki_entities = 0
    n_restricted_triples = 0
    n_restricted_group_removed = 0
    meta_relations_count = defaultdict(lambda : 0)
    meta_relations_count_all = defaultdict(lambda : 0)
    n_triples_filtered_because_of_meta = 0
    n_triples = 0
    temp_rel_pit_deprecation_count = defaultdict(lambda : 0)
    n_temp_rel_pit_deprecation_all = 0
    n_temp_rel_pit_deprecation = 0
    temp_rel_set_deprecation_count = defaultdict(lambda : 0)
    n_temp_rel_set_deprecation = 0
    n_triples_postproc = 0
    n_irrelevant_entities = 0
    n_novalue = 0
    n_somevalue = 0
    n_novalue_somevalue_group = 0
    n_corrections = 0
    n_corrections_all = 0
    validity_count = defaultdict(lambda  : 0)
    functional_count = defaultdict(lambda : 0)
    n_irrelevant_entities_triples = 0
    n_all_valid = 0
    n_empty_entities = 0
    pit_status_count = defaultdict(lambda : 0)
    n_irrelevant_entities_groups = 0

    n_relevant_entities = 0
    n_relevant_entities_triples = 0
    n_relevant_entities_groups = 0

    n_non_uniques_removed = 0
    n_non_uniques_removed_all = 0

    post_datatype_count = defaultdict(lambda : 0)
    datatype_count = defaultdict(lambda : 0)

    unique_property_object_couples = 0



    @staticmethod
    def save(path : str):
        d = {k : (dict(v) if isinstance(v,defaultdict) else v) for k,v in vars(GlobalStat).items() if isinstance(v, (int, defaultdict))}
        with open(path, 'w') as f:
            json.dump(d, f, indent=4)


    @staticmethod
    def new_type(type : str):
        GlobalStat.type_count[type] += 1
    
    @staticmethod
    def new_relation(relation : str, n : int):
        GlobalStat.relation_count[relation] += 1
        GlobalStat.relation_count_all[relation] += n

    @staticmethod
    def new_wiki_entity():
        GlobalStat.n_wiki_entities += 1

    @staticmethod
    def new_triples(n : int = 1):
        GlobalStat.n_triples += n

    @staticmethod
    def new_triples_postproc(n : int = 1):
        GlobalStat.n_triples_postproc += n
    
    @staticmethod
    def new_meta_relation(meta_rel : str, n : int):
        GlobalStat.meta_relations_count_all[meta_rel] += n
        GlobalStat.meta_relations_count[meta_rel] += 1
        GlobalStat.n_triples_filtered_because_of_meta += n

    @staticmethod
    def new_restricted_triple(qualifier : str):
        GlobalStat.n_restricted_triples += 1
        GlobalStat.n_restricted_triples_per_qualifier[qualifier] += 1

    @staticmethod
    def new_deprecated_point_in_time(relation : str):
        GlobalStat.n_temp_rel_pit_deprecation += 1
        GlobalStat.temp_rel_pit_deprecation_count[relation] += 1

    @staticmethod
    def new_deprecated_start_end_time(relation : str):
        GlobalStat.n_temp_rel_set_deprecation += 1
        GlobalStat.n_temp_rel_set_deprecation[relation] += 1

    @staticmethod
    def new_irrelevant_entity(types : list[str]):
        GlobalStat.n_irrelevant_entities += 1
        for t in types:
            GlobalStat.irrelevant_entities_per_type[t] += 1



class IrrelevantWikiEntDetector:
    def __init__(self, version : str) -> None:
        self.irrelevant = set()
        self.relevant = set()
        self.relevant_entities = set()
        self.irrelevant_entities = set()
        self.version = version

    @staticmethod
    def get_types(ent_dict : dict):
        P31s = ent_dict['claims'].get('P31', [])
        types = []
        for x in P31s:
            snak = x['mainsnak']
            if snak['snaktype'] == 'value':
                types.append(snak['datavalue']['value']['id'])
            else:
                types.append(snak['snaktype'])
        return types
    
    def classify(self, ent_id : str, types : list[str]):
        if ent_id in self.relevant_entities:
            return False
        elif ent_id in self.irrelevant_entities:
            return True
        if types is None:
            types = get_info_wikidata(ent_id, version=self.version, return_types_ids=True).get('types_ids', [])
        if len(types) == 0:
            return False
        for t in types:
            if t in self.irrelevant:
                self.irrelevant_entities.add(ent_id)
                return True
        unk_types = [t for t in types if t not in self.irrelevant and t not in self.relevant]
        infos = get_info_wikidata(unk_types, version=self.version)
        for rel_id, info in infos.items():
            name = info['name']
            if 'Wikimedia' in name or 'template' in name:
                self.irrelevant.add(rel_id)
            else:
                self.relevant.add(rel_id)
        c = any(t in self.irrelevant for t in types)
        if c:
            self.irrelevant_entities.add(ent_id)
        else:
            self.relevant_entities.add(ent_id)
        return c

class MetaPropertiesDetector:
    META_PROPERTIES_ENTITY_FILTER = 'Q51118821'
    def __init__(self, version : str) -> None:
        self.not_meta = set()
        self.meta = set()
        self.version = version

    def classify(self, prop_id : str):
        if prop_id in self.not_meta:
            return False
        if prop_id in self.meta:
            return True
        
        is_meta = self.hard_check_version(prop_id)
        if is_meta:
            self.meta.add(prop_id)
            return True
        else:
            self.not_meta.add(prop_id)
            return False
    
    def hard_check_version(self, prop_id : str):
        return MetaPropertiesDetector.META_PROPERTIES_ENTITY_FILTER in get_direct_indirect_types(prop_id, version=self.version, depth=4)


def default_to_frozen(d):
    if isinstance(d, dict):
        d = frozendict({k: default_to_frozen(v) for k, v in d.items()})
    if isinstance(d, list):
        d = tuple(default_to_frozen(x) for x in d)
    return d

class StatementTransformer:
    point_in_time_prop_id = 'P585'
    start_time_prop_id = 'P580'
    end_time_prop_id = 'P582'
    single_valued_properties = get_single_valued_properties(return_separator=True)
    temporal_single_value_properties = [k for k,v in single_valued_properties.items() if 'P585' in v]

    def __init__(self, version : str) -> None:
        self.version = version
        self.reference_timestamp = new_wikidata_timestamp if version == 'new' else old_wikidata_timestamp

    def is_deprecated_end_start_time(self, statement : dict):
        start_time, end_time = get_start_end_time(statement)
        res = start_time is not None and start_time > self.reference_timestamp or end_time is not None and  end_time < self.reference_timestamp
        GlobalStat.n_temp_rel_set_deprecation += int(res)
        return res

    def collect_validity_indicators(self, prop_id : str, statement : dict):
        d = {}
        d['is_functional'] = prop_id in self.single_valued_properties 
        start_time, end_time = get_start_end_time(statement)
        point_in_time = get_point_in_time(statement)
        d['is_temporal'] = prop_id in self.temporal_single_value_properties

        if start_time is not None and start_time < self.reference_timestamp and end_time is not None and end_time > self.reference_timestamp:
            d['is_valid'] = 'yes'
        elif end_time is None and start_time is not None and start_time < self.reference_timestamp:
            d['is_valid'] = 'yes'
        elif start_time is None and end_time is not None and end_time > self.reference_timestamp:
            d['is_valid'] = 'yes'
        elif start_time is not None and start_time > self.reference_timestamp or end_time is not None and  end_time < self.reference_timestamp:
            d['is_valid'] = 'no'
        else:
            d['is_valid'] = 'unk'

        if point_in_time is not None:
            if point_in_time < old_wikidata_timestamp:
                d['pit_status'] = 'old'
            else:
                d['pit_status'] = 'new'

        features = {}
        for name, value in [('point_in_time', point_in_time), ('start_time', point_in_time), ('end_time', point_in_time)]:
            for date, timestamp in [('old', old_wikidata_timestamp), ('new', new_wikidata_timestamp)]:
                if value is not None:
                    features[name + ' > ' + date] = value > timestamp
                    features[name + ' < ' + date] = value < timestamp
                else:
                    features[name + ' > ' + date] = False
                    features[name + ' < ' + date] = False
        statement['validity'] = d

    def remove_deprecated(self, prop_id : str, statements : list):
        # # Remove deprecated triples using qualifiers 'start time' and 'end time'
        # statements[:] = [s for s in statements if not self.is_deprecated_end_start_time(s)]

        # Keep the most up-to-date triple in temporal functional relations
        if self.point_in_time_prop_id in self.single_valued_properties.get(prop_id, []):
            point_in_times_ = [(i, get_point_in_time(statement)) for i, statement in enumerate(statements)]
            point_in_times = [(i, x) for i,x in point_in_times_ if x is not None]
            if len(point_in_times):
                GlobalStat.n_temp_rel_pit_deprecation += 1
                GlobalStat.n_temp_rel_pit_deprecation_all += len(point_in_times_)-1
                max_idx, _= max(point_in_times, key=lambda x : x[1])
                utd_statement = statements[max_idx]
                statements.clear()
                statements.append(utd_statement)
                return
    
    def is_non_unique(self, statements : list[dict]) -> bool:
        values = [default_to_frozen(statement['mainsnak']['datavalue']) for statement in statements]
        return len(values) > len(set(values))
    
    def remove_non_unique(self, statements : list):
        is_non_unique = self.is_non_unique(statements)
        if is_non_unique:
            GlobalStat.n_non_uniques_removed += 1
            GlobalStat.n_non_uniques_removed_all += len(statements)
            statements.clear()
        
    def correct_svp_errors(self, prop_id : str, statements : list):
        if prop_id in self.single_valued_properties and len(statements) != 1:
            pref = [s for s in statements if s['rank'] == 'preferred']
            if len(pref) == 1:
                GlobalStat.n_corrections += 1
                GlobalStat.n_corrections_all += len(statements)
                statements.clear()
                statements.append(pref[0])
                GlobalStat.n_corrections_all -= 1
                

class RestrictedTripleDetector:
    # review score by (P447)
    # point in time (P585)
    # start ime (P580)
    # end time (P582)
    other_restrictive_qualifiers = ['P447']
    important_qualifier = ['P585', 'P580', 'P582']
    def __init__(self, version : str) -> None:
        # self.restrictive_qualifiers = set(y for k, x in get_restrictive_qualifiers(version).items() for y in x if k != 'soft')
        self.yes = set(RestrictedTripleDetector.other_restrictive_qualifiers)
        self.no = set(RestrictedTripleDetector.important_qualifier)
        self.version = version

    def is_restricted(self, statement : dict):
        qualifiers = statement.get('qualifiers', {})
        for q_prop_id in qualifiers:
            if q_prop_id in self.yes:
                GlobalStat.n_restricted_triples += 1
                return True
            if q_prop_id in self.no:
                continue
            else:
                return self.check(q_prop_id)
        return False
    
    def check(self, prop_id : str) -> bool:
        types = get_direct_indirect_types(prop_id, self.version, depth=4)
        # If qualifier is a hard restrictive (r-instance of restrictive qualifier and not r-instance of non-restrictive qualifier) ('r' means recursive)
        if 'Q61719275' in types and 'Q61719274' not in types :
            self.yes.add(prop_id)
            return True
        # If qualifier is a social media ID
        if 'Q105388954' in types:
            self.yes.add(prop_id)
            return True
        
        self.no.add(prop_id)
        return False
        
    
def has_special_value(statement : dict):
    snaktype = statement['mainsnak']['snaktype']
    GlobalStat.n_novalue += int(snaktype == 'novalue')
    GlobalStat.n_somevalue += int(snaktype == 'somevalue')
    return snaktype != 'value'

def triple_count(ent_dict : dict):
    return sum(len(x) for x in ent_dict['claims'].values())

def group_count(ent_dict : dict):
    return len(ent_dict['claims'])

def post_analysis_stat(ent_dict):
    GlobalStat.new_triples_postproc(triple_count(ent_dict))
    GlobalStat.post_n_wiki_entities += 1
    for t in IrrelevantWikiEntDetector.get_types(ent_dict):
        GlobalStat.post_type_count[t] += 1
    for prop_id, statements in ent_dict['claims'].items():
        GlobalStat.post_relation_count[prop_id] += 1
        GlobalStat.post_relation_count_all[prop_id] += len(statements)
        for statement in statements:
            mainsnak = statement['mainsnak']
            snaktype = mainsnak['snaktype']
            if snaktype != 'value':
                continue
            GlobalStat.post_datatype_count[mainsnak['datatype']] += 1
            # property_object_counts[(prop_id, default_to_frozen(mainsnak['datavalue']['value']))] += 1
            # GlobalStat.unique_property_object_couples = len(property_object_counts)

def pre_analysis_stat(ent_dict):
    GlobalStat.new_triples(triple_count(ent_dict))
    GlobalStat.n_wiki_entities += 1
    for t in IrrelevantWikiEntDetector.get_types(ent_dict):
        GlobalStat.type_count[t] += 1
    for prop_id, statements in ent_dict['claims'].items():
        GlobalStat.relation_count[prop_id] += 1
        GlobalStat.relation_count_all[prop_id] += len(statements)
        for statement in statements:
            mainsnak = statement['mainsnak']
            snaktype = mainsnak['snaktype']
            if snaktype != 'value':
                continue
            GlobalStat.datatype_count[mainsnak['datatype']] += 1




def main(version : str):
    meta_prop_det = MetaPropertiesDetector(version)
    restricted_triplet_det = RestrictedTripleDetector(version)
    statement_transformer = StatementTransformer(version)
    irrelevant_wikient_det = IrrelevantWikiEntDetector(version)
    coll_name = 'wikidata_%s_json' % version

    def test_for_irrelevancy_triple(statement : dict):
        try:
            obj_ent_id = statement['mainsnak']['datavalue']['value']['id']
        except (KeyError, TypeError):
            return False
        if obj_ent_id.startswith('L'):
            return False
        return irrelevant_wikient_det.classify(obj_ent_id, None)
        
    
    
    to_push = []
    coll = db['wikidata_%s_prep' % version]
    coll.drop()

    # Filtering in order:
    # - Wikipedia entities only
    # - Irrelevant Wikipedia entities removal
    # - Remove meta relations
    # - special values removal
    # - Restricted triples removal
    # - Remove deprecated using qualifiers point in time
    # - Remove SVP with errors (more than one)
    # - Collect validity indicators

    # Keep only Wikipedia entities
    cursor = db[coll_name].find({'enwiki_title' : {'$exists' : 1}})
        

    for i, ent_dict in enumerate(cursor, 1):
        if _DEBUG_NUM_ENT > 0 and i > _DEBUG_NUM_ENT:
            break
        pre_analysis_stat(ent_dict)

        if irrelevant_wikient_det.classify(ent_dict['_id'], irrelevant_wikient_det.get_types(ent_dict)):
            GlobalStat.n_irrelevant_entities += 1
            GlobalStat.n_irrelevant_entities_triples += triple_count(ent_dict)
            GlobalStat.n_irrelevant_entities_groups += group_count(ent_dict)
            continue

        GlobalStat.n_relevant_entities += 1
        GlobalStat.n_relevant_entities_triples += triple_count(ent_dict)
        GlobalStat.n_relevant_entities_groups += group_count(ent_dict)

        claims = ent_dict['claims']
        for prop_id, statements in list(claims.items()):
            # # Remove triples with irrelevant objects
            # s_bef = len(statements)
            # statements[:] = [x for x in statements if test_for_irrelevancy_triple(x)]
            # GlobalStat.n_irrelevant_entities_triples += len(statements) - s_bef 
            # GlobalStat.n_relevant_entities_triples -= len(statements) - s_bef
            # if len(statements) == 0:
            #     GlobalStat.n_irrelevant_entities_groups += 1
            #     GlobalStat.n_relevant_entities_groups -= 1
            # Remove meta relations
            if meta_prop_det.classify(prop_id):
                GlobalStat.n_triples_filtered_because_of_meta += len(statements)
                GlobalStat.meta_relations_count[prop_id] += 1
                GlobalStat.meta_relations_count_all[prop_id] += len(statements)
                claims.pop(prop_id)
                continue

            # Remove triples with special value 'novalue' or 'somevalue'
            group_empty = len(statements) == 0
            statements[:] = [statement for statement in statements if not has_special_value(statement)]
            if not group_empty and len(statements) == 0:
                GlobalStat.n_novalue_somevalue_group += 1

            # Remove restricted triples
            group_empty = len(statements) == 0
            statements[:] = [statement for statement in statements if not restricted_triplet_det.is_restricted(statement)]
            if not group_empty and len(statements) == 0:
                GlobalStat.n_restricted_group_removed += 1

            # Remove deprecated statements (point in time in temporal functional relations)
            statement_transformer.remove_deprecated(prop_id, statements)

            # Remove groups with non-unique values
            # statement_transformer.remove_non_unique(statements)

            # Detect and remove single-valued errors
            statement_transformer.correct_svp_errors(prop_id, statements)

            if len(statements) == 0:
                claims.pop(prop_id)
                continue

            # Assign up-to-date label
            all_valid = True
            for statement in statements:
                statement_transformer.collect_validity_indicators(prop_id, statement)
                all_valid &= statement['validity']['is_valid'] == 'yes'
                GlobalStat.validity_count[statement['validity']['is_valid']] += 1
                GlobalStat.functional_count[statement['validity']['is_functional']] += 1
                GlobalStat.pit_status_count[statement['validity'].get('pit_status', 'none')] += 1
            GlobalStat.n_all_valid += int(all_valid) 

        if len(claims) == 0:
            GlobalStat.n_empty_entities += 1
            continue
        post_analysis_stat(ent_dict)
        to_push.append(ent_dict)

        if i % 1000 == 0:
            # break
            coll.insert_many(to_push, ordered=False)
            to_push.clear()
            print('%s/~%s (%0.4f%%)' % (i,10**7, i/10**7*100))
            GlobalStat.save(osp.join(STORAGE_FOLDER, 'resources/script_stats/preprocess_dump_%s.json' % version))

    if len(to_push):
        coll.insert_many(to_push, ordered=False)
        to_push.clear()
        print('%s/~%s (%0.4f%%)' % (i,10**7, i/10**7*100))
    GlobalStat.save(osp.join(STORAGE_FOLDER, 'resources/script_stats/preprocess_dump_%s.json' % version))
    os.makedirs(osp.join(STORAGE_FOLDER,'resources/property_lists'), exist_ok=True)
    with open(osp.join(STORAGE_FOLDER, 'resources/property_lists/post_preproc_properties_%s.json' % version), 'w') as f:
        json.dump(list(GlobalStat.post_relation_count.keys()), f) 
        
    with open(osp.join(STORAGE_FOLDER, 'resources/property_lists/restrictive_qualifiers_%s.json' % version), 'w') as f:
        json.dump(list(restricted_triplet_det.yes), f)

    with open(osp.join(STORAGE_FOLDER, 'resources/property_lists/meta_properties_%s.json' % version), 'w') as f:
        json.dump(list(GlobalStat.meta_relations_count.keys()), f)

    # with open(osp.join(STORAGE_FOLDER, '../resources/knn/prop_obj_counts_%s.json' % version), 'w') as f:
    #     json.dump(list(property_object_counts.items()), f)
    os.makedirs(osp.join(STORAGE_FOLDER,'resources/entity_lists'), exist_ok=True)
    with open(osp.join(STORAGE_FOLDER, 'resources/entity_lists/irrelevant_entities_%s.json' % version), 'w') as f:
        json.dump(list(irrelevant_wikient_det.irrelevant_entities), f)


    # print('Post-proc Relations:')
    # for k,v in get_info_wikidata(list(GlobalStat.post_relation_count)).items():
    #     print('%s (%s)' % (v['name'], k))

    # print('Restrictive qualifiers:')
    # for k,v in get_info_wikidata(list(restricted_triplet_det.yes)).items():
    #     print('%s (%s)' % (v['name'], k))

    # print('Meta relations:')
    # for k,v in get_info_wikidata(list(GlobalStat.meta_relations_count)).items():
    #     print('%s (%s)' % (v['name'], k))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        choices=["old", 'new'],
        help="The Wikidata collection to preprocess. Must be in ['old', 'new']",
        required=True
    )
    args = parser.parse_args()
    t1 = time.time()
    main(version=args.version)
    print('Execution time = %ssec' % (time.time()-t1))
