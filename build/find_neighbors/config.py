from build.config import STORAGE_FOLDER
import os.path as osp


EMBEDDINGS_FOLDER = osp.join(STORAGE_FOLDER, "gpt_encode_entity_labels")
TFIDF_INDEX_FOLDER = osp.join(STORAGE_FOLDER, "tfidf_index")