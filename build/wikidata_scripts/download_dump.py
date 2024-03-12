import sys
import time
from build.config import OLD_WIKIDATA_DATE, NEW_WIKIDATA_DATE, STORAGE_FOLDER
import os.path as osp
import json 
import subprocess
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, help='The Wikidata version to download (old or new)', choices=['old', 'new'])
args = parser.parse_args()

version = args.version

index_path = osp.join(STORAGE_FOLDER, 'wikidata_dumps_index.json')
if not osp.exists(index_path):
    print('Wikidata Dumps Index not found !\nIt is supposed to be located in %s.\n Run build_wikidata_dumps_index.py and retry.' % index_path)
    exit(1)

index = json.load(open(index_path))
url = index.get(NEW_WIKIDATA_DATE if version == 'new' else OLD_WIKIDATA_DATE)
if url is None:
    print('The specified %s_WIKIDATA_DATE was not found in the index.' % version.upper())
    print('Check the available dates at %s' % index_path)
    exit(1)

def run_wget(url, name):
    # Run wget with subprocess, capture the output
    while True:
        exit_state = subprocess.run(' '.join(['wget', '-c', '-O', '%s.json.bz2' % osp.join(STORAGE_FOLDER,name), url]), shell=True, stdout=sys.stdout, stderr=subprocess.STDOUT)
        if exit_state.returncode == 0:
            break
        time.sleep(10)
# Create threads for each wget command
thread1 = threading.Thread(target=run_wget, args=(url, '%s_wikidata' % version))

t1 = time.time()

# Start the threads
thread1.start()

# Wait for both threads to finish
thread1.join()

t2 = time.time()

print('Download time : %s sec.' % (t2-t1))
print('Finished.')
