from typing import Union
from collections import Counter
from build.config import OLD_WIKIDATA_DATE, NEW_WIKIDATA_DATE
from datetime import datetime
import numpy as np

# Precision is the priority with these tests, not recall

new_dateproperty2text = dict(P571 = dict(name='inception'), 
                             P569 = dict(name='date of birth'),
                             P580 = dict(name='start time'), 
                             P575 = dict(name='time of discovery or invention'), 
                             P1619 = dict(name='date of official opening'), 
                             P6949 = dict(name='announcement date'),
                             P585 = dict(name='point in time'))
new_dateproperty_mult2text = dict(P577 = dict(name= 'publication date', reduce="oldest"))
end_time_property = 'P582'
start_time_property = 'P580'
point_in_time_property = 'P585'

property2reduce = {
    'oldest' : lambda x : min(x)
}

old_wikidata_timestamp = np.datetime64(datetime.strptime(OLD_WIKIDATA_DATE, '%Y%m%d'), 'D')
new_wikidata_timestamp = np.datetime64(datetime.strptime(NEW_WIKIDATA_DATE, '%Y%m%d'), 'D')

def obj_is_correct(obj : str) -> bool:
    """Is the given object in a wikidata dictionary is correct or no ?

    Args:
        obj (str): Object in wikidata dictionary
    """
    res = not obj.startswith('_:')
    return res


def invariant_entity_test(entity_dict : dict):
    diff = entity_dict['diff']
    num_not_both = len([x for x in diff if x['time'] != 1])
    if num_not_both > 0:
        test_result = 'invariant'
    else:
        test_result = 'edited'
    res = {
        'test_result' : test_result
    }
    return res

def correct_time(time : str):
    sign = time[0]
    splits = time[1:].split('-')
    if splits[1] == '00':
        splits[1] = '01'
    if splits[2] == '00T00:00:00Z':
        splits[2] = '01T00:00:00Z'
    time = sign + '-'.join(splits)
    return time[:-1]

def get_point_in_time(snak : dict):
    """snak : ent_dict['claims']['P123'][i]"""
    qualifs = snak.get('qualifiers', {})
    point_in_time = None
    if point_in_time_property in qualifs and qualifs[point_in_time_property][0]['snaktype'] == 'value':
        snak = qualifs[point_in_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            point_in_time = np.datetime64(time, 'D')
        except ValueError:
            point_in_time = None
    return point_in_time

def time_value2dt64(time_value : dict):
    # time_value : snak['datavalue']['value']
    time = correct_time(time_value['time'])
    try:
        time = np.datetime64(time, 'D')
    except ValueError:
        time = None
    return time

def get_start_end_time(w : dict):
    qualifs = w.get('qualifiers', {})
    start_time, end_time = None, None
    if end_time_property in qualifs and qualifs[end_time_property][0]['snaktype'] == 'value':
        snak = qualifs[end_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            end_time = np.datetime64(time, 'D')
        except ValueError:
            end_time = None
    if start_time_property in qualifs and qualifs[start_time_property][0]['snaktype'] == 'value':
        snak = qualifs[start_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            start_time = np.datetime64(time, 'D')
        except ValueError:
            start_time = None
    return start_time, end_time

def get_point_in_time_old_and_new(w : dict):
    qualifs = w.get('qualifiers', {})
    point_in_time = None
    if point_in_time_property in qualifs and qualifs[point_in_time_property][0]['snaktype'] == 'value':
        snak = qualifs[point_in_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            point_in_time = np.datetime64(time, 'D')
        except ValueError:
            point_in_time = None

    qualifs = w.get('qualifiers_old', {})
    point_in_time_old = None
    if point_in_time_property in qualifs and qualifs[point_in_time_property][0]['snaktype'] == 'value':
        snak = qualifs[point_in_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            point_in_time_old = np.datetime64(time, 'D')
        except ValueError:
            point_in_time_old = None
    return point_in_time, point_in_time_old

def get_start_end_time_old_and_new(w : dict):
    qualifs = w.get('qualifiers', {})
    start_time, end_time = None, None
    if end_time_property in qualifs and qualifs[end_time_property][0]['snaktype'] == 'value':
        snak = qualifs[end_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            end_time = np.datetime64(time, 'D')
        except ValueError:
            end_time = None
    if start_time_property in qualifs and qualifs[start_time_property][0]['snaktype'] == 'value':
        snak = qualifs[start_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            start_time = np.datetime64(time, 'D')
        except ValueError:
            start_time = None

    qualifs = w.get('qualifiers_old', {})
    start_time_old, end_time_old = None, None
    if end_time_property in qualifs and qualifs[end_time_property][0]['snaktype'] == 'value':
        snak = qualifs[end_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            end_time_old = np.datetime64(time, 'D')
        except ValueError:
            end_time_old = None
    if start_time_property in qualifs and qualifs[start_time_property][0]['snaktype'] == 'value':
        snak = qualifs[start_time_property][0]
        time = snak['datavalue']['value']['time']
        time = correct_time(time)
        try:
            start_time_old = np.datetime64(time, 'D')
        except ValueError:
            start_time_old = None
    return start_time, end_time, start_time_old, end_time_old



class FeatureDiff:
    must_attributes = ['ent_id', 'prop_id', 'is_old', 'obj_is_ph_new', 'ent_is_ph_new']
    def __init__(self, snak : dict, ent_id : str = None, prop_id : str = None, is_old : bool = None, ent_is_ph_new : bool= None, obj_is_ph_new : bool = None) -> None:
        # snak must be from wikidata_diff collection
        validity_new = snak.get('validity_new', None)
        validity_old = snak.get('validity_old', None)
        for suffix, validity in (('', validity_new), ('_old', validity_old)):
            if validity is not None:
                if validity['is_temporal']:
                    self.relation_type = 'temp'
                elif validity['is_functional']:
                    self.relation_type = 'func'
                else:
                    self.relation_type = 'other'
                qualifiers = validity.get('qualifiers', {None})
                if qualifiers is not None:
                    snak['qualifiers' + suffix] = qualifiers
        self.start_time, self.end_time, self.start_time_old, self.end_time_old = get_start_end_time_old_and_new(snak)
        self.point_in_time, self.point_in_time_old = get_point_in_time_old_and_new(snak)
        self.diff_flag = snak['diff_flag']
        self.ent_id = ent_id
        self.prop_id = prop_id
        self.value = snak['value']
        self.is_old = is_old
        self.obj_is_ph_new = obj_is_ph_new
        self.ent_is_ph_new = ent_is_ph_new

class Feature:
    def __init__(self, snak : dict) -> None:
        # snak must be from wikidata_new_prep or wikidata_old_prep collections
        self.start_time, self.end_time = get_start_end_time(snak)
        self.point_in_time = get_point_in_time(snak)
        



def classify_algorithm_lite(features : Union[Feature, list[Feature]], version : str):
    # WARNING : No guarantee for version='new'
    reference_timestamp = old_wikidata_timestamp if version == 'old' else new_wikidata_timestamp
    if is_single := isinstance(features, Feature):
        features = [features]
    decisions = []
    for feature in features:
        decision = 'keep'
        if feature.end_time is not None and feature.end_time < reference_timestamp:
            decision = 'ignore'
        elif feature.start_time is not None and feature.start_time > reference_timestamp:
            decision = 'ignore'
        decisions.append(decision)
    if is_single:
        return decisions[0]
    return decisions

        
        
def classify_algorithm(features : list[FeatureDiff]):
    decisions = []
    paths = []
    for feature in features:
        decision = 'unk'
        path = 'unk'
        if feature.start_time is not None and feature.end_time is not None and feature.start_time > feature.end_time:
            decision = 'unk'
            path = 'incoherence_start_end_time', 'unk'
        elif feature.relation_type in ['temp', 'func']:
            count = Counter(f.diff_flag for f in features)
            if count['old'] == 1 and count['new'] == 1 and count['both'] == 0:
                if feature.diff_flag == 'new':
                    if feature.point_in_time is not None and new_wikidata_timestamp > feature.point_in_time > old_wikidata_timestamp:
                        path = 'temp_rel', '1_old_1_new', 'is_new', 'pit_between', 'learn'
                        decision = 'learn'
                    elif feature.start_time is not None and new_wikidata_timestamp > feature.start_time > old_wikidata_timestamp:
                        if feature.end_time is None or feature.end_time is not None and feature.end_time > new_wikidata_timestamp:
                            decision = 'learn'
                            path = 'temp_rel', '1_old_1_new', 'is_new', 'start_time_between', "end_time_after_new", 'learn'
        else:
            if feature.diff_flag == 'new' and feature.point_in_time is not None and new_wikidata_timestamp > feature.point_in_time > old_wikidata_timestamp:
                path = 'other_rel', 'is_new', 'pit_between', 'learn'
                decision = 'learn'
            elif feature.end_time is not None and feature.end_time < old_wikidata_timestamp:
                path = 'other_rel', 'end < old', 'ignore'
                decision = 'ignore'
            elif feature.end_time is None and feature.start_time is not None and feature.start_time < old_wikidata_timestamp:
                path = 'other_rel', 'end_none, start < old', 'keep'
                decision = 'keep'
            elif feature.end_time is None and feature.start_time is not None and old_wikidata_timestamp < feature.start_time < new_wikidata_timestamp:
                path = 'other_rel', 'end_none, start_between', 'learn'
                decision = 'learn'
            elif feature.start_time is not None and feature.start_time > new_wikidata_timestamp:
                path = 'other_rel', 'start > new', 'ignore_future'
                decision = 'ignore'
            elif feature.start_time is not None and feature.end_time is not None and new_wikidata_timestamp > feature.start_time > old_wikidata_timestamp and new_wikidata_timestamp > feature.end_time > old_wikidata_timestamp:
                path = 'other_rel', 'old < start < new, old < end < new', 'ignore_between'
                decision = 'ignore'
            elif feature.start_time is not None and feature.end_time is not None and feature.start_time < old_wikidata_timestamp and old_wikidata_timestamp < feature.end_time < new_wikidata_timestamp:
                path = 'other_rel', 'start < old, old < end < new', 'forget'
                decision = 'forget'
            elif feature.start_time is not None and feature.end_time is not None and feature.start_time < old_wikidata_timestamp and feature.end_time > new_wikidata_timestamp:
                path = 'other_rel', 'start < old, end > new', 'keep'
                decision = 'keep'
            elif feature.end_time is not None and new_wikidata_timestamp > feature.end_time > old_wikidata_timestamp:
                path = 'other_rel', 'end_betwwen', 'forget'
                decision = 'forget'
            elif feature.end_time is not None and feature.end_time > new_wikidata_timestamp:
                path = 'other_rel', 'end > new', 'keep'
                decision = 'keep'
            elif feature.diff_flag == 'old' and feature.end_time_old is not None and feature.end_time_old < old_wikidata_timestamp:
                path = 'other_rel', 'is_old', 'end_old < old', 'ignore'
                decision = 'ignore' 
            elif feature.diff_flag == 'old' and feature.end_time_old is not None and feature.end_time_old < old_wikidata_timestamp:
                path = 'other_rel', 'is_old', 'end_old < old', 'ignore'
                decision = 'ignore' 
            elif feature.diff_flag == 'new' and feature.obj_is_ph_new:
                path = 'other_rel', 'is_new', 'obj_is_ph_new', 'learn'
                decision = 'learn' 
        decisions.append(decision)
        paths.append(path)
    for i,dec in enumerate(decisions):
        feature = features[i]
        if dec == 'unk' and feature.diff_flag == 'old' and feature.relation_type in ['func', 'temp'] and len(features) == 2 and decisions[(i + 1) % 2] == 'learn':
            paths[i] = 'second_pass_func', 'forget'
            decisions[i] = 'forget'
    return decisions, paths

def classify_algorithm_full(features : list[FeatureDiff]):
    assert all(x is not None for feature in features for x in [feature.__getattribute__(y) for y in FeatureDiff.must_attributes]), " and ".join(FeatureDiff.must_attributes) + " must be set for all features to be able to call classify_algorithm_full"
    if features[0].ent_is_ph_new:
        return ['learn'] * len(features), [('ph_new_entity', 'learn')] * len(features)
    
    if features[0].is_old:
        return ['unk'] * len(features), [('old_entity', 'unk')] * len(features)
    
    prop_id = features[0].prop_id
    # P570 : date of death, P4602 : date of burial or cremation
    if prop_id in ['P570', 'P4602']:
        t = time_value2dt64(features[0].value)
        if len(features) == 1 and features[0].diff_flag == 'new' and t is not None and new_wikidata_timestamp > t > old_wikidata_timestamp:
            return ['learn'], ['date_of_death/burial', 'learn']
        else:
            return ['unk']* len(features), [('date_of_death/burial', 'unk')] * len(features)
    
    return classify_algorithm(features)