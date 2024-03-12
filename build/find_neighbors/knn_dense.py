import json
import re
from typing import Iterable, Union
import numpy as np
from build.find_neighbors.config import EMBEDDINGS_FOLDER
from build.find_neighbors.scripts.generate_entity_label_embedding import main as main_generate_label_embedding
from build.find_neighbors.scripts.index_with_nmslib import main as main_nmslib_index

import os.path as osp
import os
import nmslib
import h5py

DEFAULT_KNN_FILES_PATH = osp.join(EMBEDDINGS_FOLDER, 'entity_labels')

def _check_if_encoding_done():
    return osp.exists(osp.join(DEFAULT_KNN_FILES_PATH, 'encoded_sentences_0.h5')) and osp.exists(osp.join(DEFAULT_KNN_FILES_PATH, 'encoded_sentences_0.jsonl'))

def _check_if_nmslib_index_exists():
    return osp.exists(osp.join(DEFAULT_KNN_FILES_PATH, 'entities.txt')) and osp.exists(osp.join(DEFAULT_KNN_FILES_PATH, 'nmslib_embeddings.idx'))



class KNearestNeighbors:
    def __init__(self, model_name="EleutherAI/gpt-neo-2.7B", version=['old', 'new']) -> None:
        """Class to setup a neareset neighbor system using NMSlib on entity labels from wikidata

        Args:
            model_name (str, optional): The huggingface model to use to encode entity labels. Defaults to "EleutherAI/gpt-neo-2.7B".
            version (list, optional): From which version of Wikidata we collect entities. The label of a given entity is first retrieved from the first version in the list then the second (if not found in the first). Defaults to ['old', 'new'].
        """
        if isinstance(version, str):
            version = [version]
        assert all(v in ['old', 'new'] for v in version)
        self.version = version
        self.model_name = model_name
        self.index = None
        self.entities = None

    def setup(self, reset=False):
        if reset or not _check_if_encoding_done():
            print('Entity labels embeddings NOT detected.')
            print('Embedding Generation launched.')
            print('IMPORTANT : If the progress bar is stucked, it means that no GPU were available when the script was launched or that something went wrong.')
            print('You can restart the program if this happens. Progress is saved.')
            for v in self.version:
                main_generate_label_embedding("entity_labels", self.model_name, version=v)

        else:
            print('Entity labels embeddings detected.')
        
        if reset or not _check_if_nmslib_index_exists():
            print('NMSlib NOT detected.')
            print('NMSlib index creation launched.')
            main_nmslib_index(DEFAULT_KNN_FILES_PATH)

        print('Environment for KNearestNeighbors setuped!')

    def load_index(self):
        if not _check_if_nmslib_index_exists():
            raise Exception('Error : Index does not exist. Call the setup() function to create it.')
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        print('Loading Index...')
        self.index.loadIndex(osp.join(DEFAULT_KNN_FILES_PATH, 'nmslib_embeddings.idx'))
        print('Load entities IDs...')
        with open(osp.join(DEFAULT_KNN_FILES_PATH, 'entities.txt')) as f:
            self.entities = [x.strip() for x in f.readlines()]
        self.l_dsets = []
        self.l_id2int = []
        self.h5_files = []
        for x in os.listdir(DEFAULT_KNN_FILES_PATH):
            if m := re.match(r"^encoded_sentences_([0-9]+)\.h5$", x):
                print('Reading %s...' % x)
                with open(osp.join(DEFAULT_KNN_FILES_PATH, "encoded_sentences_%s.jsonl" % m.group(1))) as f:
                    self.l_id2int.append({json.loads(x)['_id'] : i for i,x in enumerate(f.readlines())})
                f = h5py.File(osp.join(DEFAULT_KNN_FILES_PATH, x), 'r')
                dset = f['encodings']
                self.l_dsets.append(dset)
                self.h5_files.append(f)
        
    
    def get_vectors(self, ids : Union[str, Iterable[str]]) -> Union[np.ndarray, list[np.ndarray]]:
        if is_single := isinstance(ids, str):
            ids = [ids]
        vectors = []
        for ent_id in ids:
            for i, id2int in enumerate(self.l_id2int):
                idx = id2int.get(ent_id, None)
                if idx is not None:
                    break
            else:
                print('WARNING : %s ID not found in the index' % ent_id)
                return None
            
            vector = self.l_dsets[i][idx]
            vectors.append(vector)
        if is_single:
            return vectors[0]
        return vectors

    def get_nearest(self, emb_vector : list[np.ndarray], k=10):
        if self.index is None:
            raise Exception('Error : Index not loaded. Load the index using KNearestNeighbors.load_index() function.')
        ids, distances = self.index.knnQuery(emb_vector, k=k)
        ids = [self.entities[i] for i in ids]
        return ids, distances

