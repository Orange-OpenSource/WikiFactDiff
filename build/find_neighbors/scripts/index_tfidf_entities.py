import sys
from build.utils.wd import db, ent_claims_to_vec
from build.gpt3_5_verbalization.verbalize_wikidata import run_in_process
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import nmslib
import scipy.io
from build.find_neighbors.config import TFIDF_INDEX_FOLDER
import os.path as osp
import shutil
import os
from joblib import dump

@run_in_process
def generator(version : str, seen_entities : set[str]):
    total = db['wikidata_%s_prep' % version].estimated_document_count()
    seen_entities = set(seen_entities)

    for x in tqdm.tqdm(db['wikidata_%s_prep' % version].aggregate([
        {'$project' : {'_id' : 1}},
        {'$lookup' : {
            'from' : "wikidata_%s_json" % version,
            'localField' : "_id",
            'foreignField' : "_id",
            'as' : 'wd'
        }},
        {'$replaceRoot' : {"newRoot" : {"$first" : "$wd"}}},
        {"$project" : {"claims" : 1}},
        # {"$limit" : 1000}
    ]), total=total, mininterval=5, file=sys.stdout):
        if x['_id'] in seen_entities:
            continue
        yield x['_id'], x['claims']
        
def save_list(l : list[str], path : str):
    with open(path, 'w') as f:
        for x in l:
            f.write(x + '\n')

def identity(x):
    return x
        
def main():
    # Reset environment
    shutil.rmtree(TFIDF_INDEX_FOLDER, ignore_errors=True)
    os.mkdir(TFIDF_INDEX_FOLDER)

    seen_entities = set()
    list_seen_entities = []
    ent_vectors = []
    version_count = [0,0]
    print('Collecting entity features (relation-object couples, relations, and objects)...')
    for version_i, version in enumerate(['old', 'new']):
        print('From Wikidata %s' % version)
        for ent_id, claims in generator(version, list(seen_entities)):
            seen_entities.add(ent_id)
            list_seen_entities.append(ent_id)
            ent_vec = ent_claims_to_vec(claims, add_singles=True)
            # Add Entity ID. Scenario : This ID appears in another vector and indicates these two entities are linked
            ent_vec.append(ent_id)
            ent_vectors.append(ent_vec)
            version_count[version_i] += 1
    
    tfidf = TfidfVectorizer(analyzer=identity, norm=None,)

    print('Building TF-IDF vectors...')
    features_sparses = tfidf.fit_transform(ent_vectors)
    print('TF-IDF Matrix Shape : %s' % str(features_sparses.shape))
    dump(tfidf, osp.join(TFIDF_INDEX_FOLDER, "tfidf_vectorizer.pkl"))

    

    features_sparses_old, features_sparses_new = features_sparses[:version_count[0]], features_sparses[version_count[0]:]
    entities_old, entities_new = list_seen_entities[:version_count[0]], list_seen_entities[version_count[0]:]
    save_list(entities_old, osp.join(TFIDF_INDEX_FOLDER, 'entities_old.txt'))
    save_list(entities_new, osp.join(TFIDF_INDEX_FOLDER, 'entities_new.txt'))
    scipy.sparse.save_npz(osp.join(TFIDF_INDEX_FOLDER, 'features_sparses_old.npz'), features_sparses_old)
    scipy.sparse.save_npz(osp.join(TFIDF_INDEX_FOLDER, 'features_sparses_new.npz'), features_sparses_new)

    print('Index creation...')
    # Initialize NMSLib index
    index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)

    # Add data points to the index
    index.addDataPointBatch(features_sparses_old)

    # Create the index
    index.createIndex({'post': 2}, print_progress=True)
    print("\nSave index...")
    index.saveIndex(osp.join(TFIDF_INDEX_FOLDER, 'index.bin'), save_data=True)
    print('Finished.')


if __name__ == '__main__':
    main()
