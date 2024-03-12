from typing import Iterable, Union
import numpy as np
from build.find_neighbors.config import TFIDF_INDEX_FOLDER
from build.find_neighbors.scripts.index_tfidf_entities import main as main_nmslib_index

import os.path as osp
import nmslib
import scipy.sparse


def _check_if_nmslib_index_exists():
    return osp.exists(osp.join(TFIDF_INDEX_FOLDER, 'index.bin'))

def read_list(path : str) -> list[str]:
    with open(path, 'r') as f:
        l = [x.strip() for x in f]
    return l


class KNearestNeighbors:
    def __init__(self) -> None:
        """Class to setup a neareset neighbor system using NMSlib on entity labels from wikidata.
        Each entity is encoded into a sparse TF-IDF vector
        """
        self.index = None
        self.entities_in_index = None
        self.entities_new = None
        self.entities_in_index_name2idx = None
        self.entities_new_name2idx = None
        self.tfidf_old = None
        self.tfidf_new = None

    def setup(self, reset=False):        
        if reset or not _check_if_nmslib_index_exists():
            print('NMSlib index NOT detected.')
            print('NMSlib index creation launched.')
            main_nmslib_index()

        print('Environment for KNearestNeighbors setuped!')

    def load_index(self):
        if not _check_if_nmslib_index_exists():
            raise Exception('Error : Index does not exist. Call the setup() function to create it.')
        self.index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
        print('Loading Index...')
        self.index.loadIndex(osp.join(TFIDF_INDEX_FOLDER, 'index.bin'), load_data=True)
        self.index.setQueryTimeParams({'efSearch': 2000})

        print('Loading entity names...')
        self.entities_in_index = read_list(osp.join(TFIDF_INDEX_FOLDER, 'entities_old.txt'))
        self.entities_new = read_list(osp.join(TFIDF_INDEX_FOLDER, 'entities_new.txt'))
        self.entities_in_index_name2idx = { k : i for i,k in enumerate(self.entities_in_index)}
        self.entities_new_name2idx = { k : i for i,k in enumerate(self.entities_new)}

        print('Loading TF-IDF sparse matrix...')
        self.tfidf_old = scipy.sparse.load_npz(osp.join(TFIDF_INDEX_FOLDER,'features_sparses_old.npz'))
        self.tfidf_new = scipy.sparse.load_npz(osp.join(TFIDF_INDEX_FOLDER,'features_sparses_new.npz'))

    
    def get_vectors(self, ids : Union[str, Iterable[str]]) -> Union[np.ndarray, list[np.ndarray]]:
        if is_single := isinstance(ids, str):
            ids = [ids]
        vectors = []
        for ent_id in ids:
            for version, id2int in (('old', self.entities_in_index_name2idx), ('new', self.entities_new_name2idx)):
                idx = id2int.get(ent_id, None)
                if idx is not None:
                    break
            else:
                print('WARNING : %s ID not found in the index' % ent_id)
                return None
            if version == 'old':
                vector = self.tfidf_old[idx]
            else:
                vector = self.tfidf_new[idx]
            vectors.append(vector)
        if is_single:
            return vectors[0]
        return vectors

    def get_nearest(self, emb_vector : list[np.ndarray], k=10):
        if self.index is None:
            raise Exception('Error : Index not loaded. Load the index using KNearestNeighbors.load_index() function.')
        ids, distances = self.index.knnQueryBatch(emb_vector, k=k)[0]
        ids = [self.entities_in_index[i] for i in ids]
        return ids, distances

if __name__ == '__main__':
    from utils import get_info_wikidata
    knn = KNearestNeighbors()
    knn.load_index()
    print([get_info_wikidata(x.strip())['name'] for x in knn.get_nearest(knn.get_vectors('Q142'), k=100)[0]])
    print("hhhh")