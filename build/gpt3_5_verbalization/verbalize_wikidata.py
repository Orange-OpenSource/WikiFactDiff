from collections import Counter
from build.gpt3_5_verbalization.utils import (
    parallel_verbalize,
    Entity,
    KnowledgeTriple,
    Literal,
    Property,
)
from build.utils.wd import db, get_info_wikidata, get_value

import threading
import queue
from functools import wraps
import json
import os.path as osp
import random
import time
from multiprocessing import Process, Queue
from functools import wraps

from build.config import STORAGE_FOLDER



RESUME = True
WHAT_TO_VERBALIZE = "wfd"  # Can take two values ['old_wikidata', 'wfd']


def run_in_process(gen_func):
    @wraps(gen_func)
    def wrapper(*args, **kwargs):
        queue = Queue(100)

        def target_func(queue: Queue, *args, **kwargs):
            gen = gen_func(*args, **kwargs)
            for item in gen:
                queue.put(item)
            queue.put(StopIteration)

        p = Process(target=target_func, args=(queue,) + args, kwargs=kwargs)
        p.start()

        while True:
            item = queue.get()
            if item is StopIteration:
                break
            yield item

        p.join()

    return wrapper


SAVE_FOLDER = osp.join(STORAGE_FOLDER, "chatgpt_wkd_verbalization")


class Finish(Exception):
    pass


class PropertyInfoRetriever:
    def __init__(self, versions: list = ["old"]) -> None:
        self.property_infos = {}
        self.property_examples = {}
        self.versions = versions

    def retrieve(self, prop_id: str) -> dict:
        if prop_id in self.property_infos:
            return self.property_infos[prop_id]
        for version in self.versions:
            prop_info = get_info_wikidata(prop_id, version=version)
            if len(prop_info):
                self.property_infos[prop_id] = prop_info
                break
        else:
            self.property_infos[prop_id] = None
        return prop_info

    def retrieve_example(self, prop: Property, version="old"):
        if prop.id in self.property_examples:
            return self.property_examples[prop.id]

        sub_obj = db["wikidata_%s_json" % version].find_one(
            {"_id": prop.id},
            {
                "subject": "$claims.P1855.mainsnak.datavalue.value.id",
                "object": "$claims.P1855.qualifiers.%s.datavalue.value" % prop.id,
            },
        )
        if (
            sub_obj is None
            or "object" not in sub_obj
            or len(sub_obj["subject"]) == 0
            or len(sub_obj["object"]) == 0
        ):
            # No examples found
            with open(osp.join(SAVE_FOLDER, "no_example_property.txt"), "a") as f:
                f.write(prop.id + "\n")
            self.property_examples[prop.id] = None
            return
        sub_info = get_info_wikidata(sub_obj["subject"][0], version=version)
        subject = Entity(sub_info["id"], sub_info["name"], sub_info["description"])
        obj_raw = sub_obj["object"][0][0]
        if isinstance(obj_raw, dict) and "id" in obj_raw:
            obj_info = get_info_wikidata(obj_raw["id"], version=version)
            object = Entity(obj_info["id"], obj_info["name"], obj_info["description"])
        else:
            object = Literal(*get_value(obj_raw, add_unit=True, add_type=True))

        example = KnowledgeTriple(subject, prop, object)
        self.property_examples[prop.id] = example
        return example


def convert_to_triple_for_chat_gpt_verb(
    prop_info_ret: PropertyInfoRetriever,
    ent_info: dict,
    prop_id: str,
    values: list[dict],
) -> KnowledgeTriple:
    random.shuffle(values)
    prop_info = prop_info_ret.retrieve(prop_id)

    ent = Entity(ent_info["id"], ent_info["name"], ent_info["description"])
    prop = Property(prop_id, prop_info["name"], prop_info["description"])
    for value in values:
        obj, t = get_value(
            value, get_info_dict_for_entities=True, add_type=True, add_unit=True
        )
        if obj is None:
            continue
        if isinstance(obj, dict):
            if len(obj) == 0:
                obj, t = get_value(
                    value,
                    get_info_dict_for_entities=True,
                    version="old",
                    add_type=True,
                    add_unit=True,
                )
            if len(obj) == 0:
                continue
            obj = Entity(obj["id"], obj["name"], obj["description"])
            break
        else:
            obj = Literal(obj, t)
        break
    else:
        return None
    triple = KnowledgeTriple(ent, prop, obj)
    return triple


def buffered_generator(buffer_size=20):
    def decorator(gen_func):
        @wraps(gen_func)
        def wrapper(*args, **kwargs):
            gen = gen_func(*args, **kwargs)
            q = queue.Queue(maxsize=buffer_size)

            # Function to run in a separate thread to populate the queue
            def worker():
                for item in gen:
                    q.put(item)

            def counter():
                while True:
                    print("Queue size = %s" % q.qsize())
                    time.sleep(0.5)

            # Start the worker thread
            t1 = threading.Thread(target=worker, daemon=True)
            # t2 = threading.Thread(target=counter, daemon=True)
            t1.start()
            # t2.start()

            # Yield items from the queue
            while True:
                item = q.get()
                yield item

        return wrapper

    return decorator


N_ENTITIES_SAMPLE = 100000
N_FOR_SUCCESS = 100


@run_in_process
def triple_generator_wfd():
    cursor = db["wkd_fd"].aggregate(
        [
            {"$sort": {"ent_imp": -1}},
        ]
    )

    print("Verbalization begins")
    for i, x in enumerate(cursor, 1):
        ent_id = x["_id"]["ent_id"]
        ent_info = get_info_wikidata(ent_id, version="new")
        prop_id = x["_id"]["prop_id"]
        # print(ent_info)

        values = [y["value"] for y in x["snaks"]]
        try:
            triple = convert_to_triple_for_chat_gpt_verb(
                prop_info_ret, ent_info, prop_id, values
            )
            if triple is None:
                continue
        except:
            print("Serious error:", ent_info, prop_id)
            continue
        triple.relation.triple_example = prop_info_ret.retrieve_example(
            triple.relation, version="new"
        )

        yield triple, {"use_support": True}

        if i % 10 == 0:
            print("N_groups = %s" % i)


@run_in_process
def triple_generator_old_wikidata():
    count_entities = 0
    cursor = db["old_entities_importance"].aggregate(
        [
            {"$sort": {"ent_imp": -1}},
            {"$project": {"_id": 1}},
            {"$limit": N_ENTITIES_SAMPLE},
        ]
    )
    print("Collecting Entities Sample...")
    entities_sample = [x["_id"] for x in cursor]
    # entities_sample = ['Q1805085']

    random.seed(12345678)
    random.shuffle(entities_sample)

    cursor = db["wikidata_old_prep"].aggregate(
        [
            {"$match": {"_id": {"$in": entities_sample}}},
            {"$addFields": {"__order": {"$indexOfArray": [entities_sample, "$_id"]}}},
            {"$sort": {"__order": 1}},
            {"$project": {"claims": 1}},
        ]
    )

    print("Verbalization begins")
    try:
        for i, x in enumerate(cursor, 1):
            count_entities += 1
            ent_id = x["_id"]
            ent_info = get_info_wikidata(ent_id, version="old")
            # print(ent_info)
            for prop_id, values in x["claims"].items():
                properties_remaining = sum(
                    int(x < N_FOR_SUCCESS) for x in prop_count.values()
                )
                if properties_remaining <= 0:
                    raise Finish

                values = [y["mainsnak"]["datavalue"]["value"] for y in values]
                if prop_count[prop_id] > N_FOR_SUCCESS:
                    continue
                try:
                    triple = convert_to_triple_for_chat_gpt_verb(
                        prop_info_ret, ent_info, prop_id, values
                    )
                    if triple is None:
                        continue
                except:
                    print("Serious error:", ent_info, prop_id)
                    continue
                triple.relation.triple_example = prop_info_ret.retrieve_example(
                    triple.relation
                )

                prop_count[triple.relation.id] += 1
                yield triple, {"use_support": True}

            if i % 5 == 0:
                print(
                    "N_entities = %s, Remaining properties = %s"
                    % (i, properties_remaining)
                )
    except Finish:
        pass


if __name__ == "__main__":
    prop_info_ret = PropertyInfoRetriever(
        versions=["old"] if WHAT_TO_VERBALIZE == "old_wikidata" else ["new"]
    )

    # Load history of prop_count
    prop_count = {
        k: 0
        for k in json.load(
            open(
                osp.join(
                    osp.dirname(__file__),
                    "../resources/script_stats/preprocess_dump_old.json",
                )
            )
        )["post_relation_count"].keys()
    }
    try:
        with open(osp.join(SAVE_FOLDER, "verbalizations.jsonl")) as f:
            relations = [
                json.loads(x)["triple"]["relation"]["id"] for x in f.readlines()
            ]
    except FileNotFoundError:
        relations = []
    if RESUME and len(relations):
        for k, v in Counter(relations).items():
            prop_count[k] = v

    gen = (
        triple_generator_old_wikidata()
        if WHAT_TO_VERBALIZE == "old_wikidata"
        else triple_generator_wfd()
    )
    n_triples = (
        db["wkd_fd"].estimated_document_count() if WHAT_TO_VERBALIZE == "wfd" else None
    )
    parallel_verbalize(
        gen, SAVE_FOLDER, n_threads=12, verbose=1, resume=RESUME, n_triples=n_triples
    )
