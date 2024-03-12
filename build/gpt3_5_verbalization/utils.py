from __future__ import annotations
import json
from typing import Iterator, Union
import re
import openai
import openai.error
from multiprocessing.pool import ThreadPool
import time
import os.path as osp
import shutil
import os

# OpenAI API config
# IMPORTANT : Set these constants to use ChatGPT to verbalize triples
OPENAI_API_TYPE = ""
OPENAI_API_BASE = ""
OPENAI_API_VERSION = ""
OPENAI_ENGINE = ""

# Prompt config
MAX_TOKENS = 800
# PRE_PROMPT = """You are an AI assistant that helps people find information."""
PRE_PROMPT = """You are an advanced knowledge triple verbalization system.
You take as input a knowledge triple (subject, relation, object) and generate a list of 10 linguistically diverse verbalizations of the triple. 
For example, the input could be : (France, capital, Paris) and one of your verbalizations may be : "The capital of France is Paris".  

The veracity of the knowledge triple does not affect the quality of your generation.

Examples of correct verbalizations:
- (Matriak, instance of, university) --> "Matriak is a university."
- (Johnathan Smith, date of death, 11-05-2012) --> "Johnathan Smith died in 11-05-2012.\"
- (Tranquility Base Hotel & Casino, follows, AM) --> "Tranquility Base Hotel & Casino follows AM.\"
- (Paris, named after, Parisii) --> "Paris was named after Parisii.\""""

MAIN_PROMPT = """Here is the knowledge triple to verbalize: (%s, %s, %s). Your sentences should be concise and end with the term %s."""

SUPPORT_PROMPT = """Due to the ambiguity that could arise from the provided labels, here is their meaning:
- (subject) "%s" : "%s"
- (relation) "%s" : "%s"
- (object) "%s" : "%s\""""


class Entity:
    def __init__(self, id: str, label:str, description:str) -> None:
        self.id = id
        self.label = label
        self.description = description

    def to_dict(self):
        return {
            'id' : self.id,
            'label' : self.label,
            'description' : self.description
        }
    
class Property(Entity):
    def __init__(self, id: str, label: str, description: str, triple_example : KnowledgeTriple = None) -> None:
        super().__init__(id, label, description)
        self.triple_example = triple_example

class Literal:
    def __init__(self, label : str, description : str) -> None:
        self.label = label
        self.description = description

    def to_dict(self) -> dict:
        return {
            'label' : self.label,
            'description' : self.description
        }

class KnowledgeTriple:
    HASH_LEVEL = "GROUP"
    def __init__(self, subject : Entity, relation : Property, object : Union[Entity, Literal]) -> None:
        self.subject = subject
        self.relation = relation
        self.object = object
    
    def get_labels(self):
        return (self.subject.label, self.relation.label, self.object.label)
    
    def to_dict(self):
        return {
            'subject' : self.subject.to_dict(),
            'relation' : self.relation.to_dict(),
            'object' : self.object.to_dict()
        }

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, KnowledgeTriple):
            return False
        if KnowledgeTriple.HASH_LEVEL != 'GROUP':
            return self.subject.id == __value.subject.id and self.relation.id == __value.relation.id and self.object.label == __value.object.label
        else:
            return self.subject.id == __value.subject.id and self.relation.id == __value.relation.id

    def __hash__(self) -> int:
        if KnowledgeTriple.HASH_LEVEL != 'GROUP':
            return hash((self.subject.id, self.relation.id, self.object.label))
        else:
            return hash((self.subject.id, self.relation.id))

def extract_verbalizations(message : str):
    try:
        verbalizations = re.findall(r"(^|\n)([0-9]+?\.|\-) ['\"]?(.+?)['\"\.]*(?=(\n|$))", message)
        verbalizations = [x[2].strip() for x in verbalizations]
    except:
        return None

    return verbalizations

class ChatGPTVerbalizer:
    def __init__(self, dummy_mode=False) -> None:
        openai.api_type = OPENAI_API_TYPE
        openai.api_base = OPENAI_API_BASE
        openai.api_version = OPENAI_API_VERSION
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if openai.api_key is None or len(openai.api_key) == 0:
            raise Exception('Error : OPENAI_API_KEY not set!')

        self.openai_config = {
            'api_type' : openai.api_type,
            'api_base' : openai.api_base,
            'api_version' : openai.api_version,
        }
        self.dummy_mode = dummy_mode
    
    def verbalize(self, triple : KnowledgeTriple, use_support : bool = False):
        if self.dummy_mode:
            return {'verbs' : ['DUMMY']*5, 'total_tokens': 0, 'chatgpt_response' : "DUMMY"}
        prompt = MAIN_PROMPT % (triple.get_labels() + (triple.object.label,))
        if use_support:
            prompt += '\n' + SUPPORT_PROMPT % (triple.subject.label, triple.subject.description,triple.relation.label, triple.relation.description,triple.object.label, triple.object.description)
        if triple.relation.triple_example is not None:
            example = triple.relation.triple_example
            prompt += ' Finally, here is an example where the relation "%s" is employed : (%s, %s, %s).' % (example.relation.label, *example.get_labels()) 
        try:
            response = openai.ChatCompletion.create(
                engine=OPENAI_ENGINE,
                messages = [{"role":"system","content":PRE_PROMPT},{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=MAX_TOKENS,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
        except (openai.error.InvalidRequestError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout) as e:
            if isinstance(e, (openai.error.InvalidRequestError,)):
                return {"error" : e.__class__.__name__}
            else:
                return None

        if response['choices'][0]['finish_reason'] == 'content_filter':
            return {"error" : "content_filter"}
        
        message = response['choices'][0]['message']['content']
        verbs = extract_verbalizations(message)
        try:
            total_tokens = response['usage']['total_tokens']
        except KeyError:
            total_tokens = None
        return {'verbs' : verbs, 'total_tokens': total_tokens, 'chatgpt_response' : message, 'error' : None}

def analyze_verbalizations(triple : KnowledgeTriple, verbalizations : dict, chatgpt_options : dict) -> dict:
    verbalizations_, total_tokens, chatgpt_response = verbalizations.get('verbs', None), verbalizations.get('total_tokens', None), verbalizations.get('chatgpt_response', None)
    verbs = []

    d = {
        'triple' : triple.to_dict(),
        'chatgpt_options' : chatgpt_options,
        'verbalizations' : verbs,
        'total_tokens' : total_tokens,
        'chatgpt_response' : chatgpt_response,
        'extraction_failed' : verbalizations_ is None,
        'error' : verbalizations['error']
    }
    if d['extraction_failed']:
        return d

    subject_label, _, object_label = triple.get_labels()
    for v in verbalizations_:
        contains_object = object_label.lower() in v.lower()
        contains_subject = subject_label.lower() in v.lower()
        search_object = re.search(rf'{re.escape(object_label)}[\.]?$', v, re.IGNORECASE)
        ends_with_object = search_object is not None
        if ends_with_object:
            fill_in_the_blank = re.sub(rf'{re.escape(object_label)}[\.]?$', r'____', v, flags=re.IGNORECASE)
        else:
            fill_in_the_blank = None
        verbs.append({
            'verbalization' : v,
            'contains_object' : contains_object,
            'contains_subject' : contains_subject,
            'ends_with_object' : ends_with_object,
            "fill_in_the_blank" : fill_in_the_blank,
        })
    return d


def _setup_env(save_folder : str, openai_config, main_prompt=MAIN_PROMPT, support_prompt=SUPPORT_PROMPT, max_tokens=MAX_TOKENS):
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    with open(osp.join(save_folder, 'config.json'), 'w') as f:
        json.dump({
            'main_promp' : main_prompt,
            'support_prompt' : support_prompt,
            'max_tokens' : max_tokens,
            'openai_config': openai_config
        }, f, indent=4)


def parallel_verbalize(triples_options_chatgpt : Iterator[tuple[KnowledgeTriple, dict]], save_folder : str, n_threads=8, n_limit=None, n_triples=None, resume=False, verbose=0) -> Iterator[dict]:
    def _thread_map(args):
        triple : KnowledgeTriple
        option_chatgpt : dict
        chatgpt : ChatGPTVerbalizer
        triple, option_chatgpt, chatgpt = args
        verbalizations = None
        while not verbalizations:
            verbalizations = chatgpt.verbalize(triple, **option_chatgpt)
            if verbalizations is None:
                time.sleep(1)
        return analyze_verbalizations(triple, verbalizations, option_chatgpt)
    
    def _generator(seen_triples : set[KnowledgeTriple]):
        for i, (t, opt) in enumerate(triples_options_chatgpt):
            if t in seen_triples:
                continue
            if n_limit is not None and i >= n_limit:
                break
            yield t, opt, chatgpts[i%n_threads]

    
    # Initialize Pool + ChatGPTs
    pool = ThreadPool(n_threads)
    chatgpts = [ChatGPTVerbalizer()]*n_threads
    
    seen_triples = set()

    # Prepare save folder
    if not resume:
        _setup_env(save_folder, chatgpts[0].openai_config)
        
    if resume:
        oai_config = chatgpts[0].openai_config
        main_prompt, support_prompt, max_tokens = MAIN_PROMPT, SUPPORT_PROMPT, MAX_TOKENS
        try:
            with open(osp.join(save_folder, 'config.json'), 'r') as f:
                config = json.load(f)
                main_prompt = config['main_promp']
                support_prompt = config['support_prompt']
                max_tokens = config['max_tokens']
                oai_config = config['openai_config']
                oai_config['api_key'] = os.getenv('OPENAI_API_KEY')
            for chatgpt in chatgpts:
                chatgpt.openai_config = oai_config
            with open(osp.join(save_folder, 'verbalizations.jsonl'), 'r') as f:
                verbs = [json.loads(x) for x in f.readlines()]
            seen_triples = [x['triple'] for x in verbs]
            build_object = lambda x : Entity(**x['object']) if len(x['object']) == 3 else Literal(**x['object'])
            seen_triples = set(KnowledgeTriple(Entity(**x['subject']), Property(**x['relation']), build_object(x)) for x in seen_triples)
        except FileNotFoundError:
            print('Could not resume verbalization :: config.json and/or verbalizations.json not found!')
            answer = input("Do you want to restart from scratch ? (Y/N)")
            if answer.lower().strip() != 'y':
                exit()
            resume = False
            _setup_env(save_folder, oai_config, main_prompt, support_prompt, max_tokens)
    file = open(osp.join(save_folder, 'verbalizations.jsonl'), 'a')
    n_triples = '?' if n_triples is None else n_triples
    for i,d in enumerate(pool.imap(_thread_map, _generator(seen_triples), chunksize=1), start=len(seen_triples)):
        if d is None:
            continue
        file.write(json.dumps(d) + '\n')
        file.flush()
        if verbose:
            if i % 5 == 0:
                print('%s/%s' % (i, n_triples))

    file.close()
