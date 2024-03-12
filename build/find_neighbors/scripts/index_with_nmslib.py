import argparse
import nmslib
import numpy
from build.find_neighbors.config import EMBEDDINGS_FOLDER
import os
import os.path as osp
import re
import json
import h5py

def save_list_in_file(l : list, file_path : str):
    with open(file_path, 'w') as f:
        for x in l:
            f.write(x + '\n')
    

def main(subfolder_path : str):
    index = nmslib.init(method='hnsw', space='cosinesimil')
    entities = []
    print('Reading entity labels files')
    for x in os.listdir(subfolder_path):
        if m := re.match(r"^encoded_sentences_([0-9]+)\.h5$", x):
            print('Reading %s...' % x)
            with open(osp.join(subfolder_path, "encoded_sentences_%s.jsonl" % m.group(1))) as f:
                entities.extend([json.loads(x)['_id'] for x in f.readlines()])
            with h5py.File(osp.join(subfolder_path, x), 'r') as f:
                dset = f['encodings']
                print('Pushing to NMSlib...')
                index.addDataPointBatch(dset)
            
    print('Save entity ids...')
    save_list_in_file(entities, osp.join(subfolder_path, 'entities.txt'))
    print('Creating NMSlib index...')
    index.createIndex({'post': 2}, print_progress=True)
    print('Saving the index...')
    index.saveIndex(osp.join(subfolder_path, 'nmslib_embeddings.idx'), save_data=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for create a NMSlib index from entity labels embeddings.")

    parser.add_argument('--subfolder_path', type=str, required=True, help="Folder that contains the entity label embeddings")
    args = parser.parse_args()
    main(args.subfolder_path)
