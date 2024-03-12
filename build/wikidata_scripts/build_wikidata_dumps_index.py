import requests
import re
from internetarchive import search_items, get_item
import json
import os
import multiprocessing as mp
from build.config import STORAGE_FOLDER

wikidata_dump_url = 'https://dumps.wikimedia.org/wikidatawiki/entities/'
pattern = r'^wikidata-[0-9]{8}-all.json.bz2$'

def process_wikidata_dump(d : str):
    # d <==> date
    html = requests.get(wikidata_dump_url + d).content.decode()
    files = [x for x in re.findall(r'<a href="(.+?)">.+?</a>', html) if re.match(pattern, x)]
    
    if len(files):
        assert len(files) == 1
        url = wikidata_dump_url + d + '/' + files[0]
        print('Added from Wikidata dump - %s : %s' % (d, url))
        return url, d
    
def process_internet_archive(identifier : str):
    date_id = identifier[-8:]
    item = get_item(identifier)
    for x in item.files:
        if re.match(pattern, x['name']):
            url = "https://archive.org/download/{}/{}".format(identifier, x['name'])
            print('Added from Internet Archive - %s : %s' % (date_id, url))
            return url, date_id 
            

if __name__ == '__main__':
    try:
        index = {}

        # Check on wikidata dumps website
        html = requests.get(wikidata_dump_url).content.decode()
        dates = [x[:-1] for x in re.findall(r'<a href="(.+?)">.+?</a>', html) if x.endswith('/') and x != '../']
        
        print('Number of potienial dumps found in Wikidata dump = %s' % len(dates))
        with mp.Pool(max(1, mp.cpu_count() // 2)) as pool:
            results = [x for x in pool.map(process_wikidata_dump, dates) if x is not None]
        for url, d in results:
            index[d] = url
        
        # Check on Internet Archive
        identifiers = search_items('creator:"Wikidata editors" AND mediatype:"web" AND collection:"wikicollections" AND title:"Wikidata entity dumps (JSON and TTL) of all Wikibase entries for Wikidata generated on"')
        identifiers = [x['identifier'] for x in identifiers]
        print('Number of potienial dumps found in Internet Archive = %s' % len(identifiers))
        with mp.Pool(max(1, mp.cpu_count() // 2)) as pool:
            results = [x for x in pool.map(process_internet_archive, identifiers) if x is not None]
        for url, d in results:
            index[d] = url
        
        print('\n\nAll availables JSON dumps:')
        print('==========================\n')
        for x in sorted(index):
            print(x)
        print()
        print('Total number of JSON dumps found = %s' % len(index))
        os.makedirs(STORAGE_FOLDER, exist_ok=True)
        json_path = os.path.join(STORAGE_FOLDER, 'wikidata_dumps_index.json')
        with open(json_path, 'w') as f:
            json.dump(index, f)
        print('Finished.')
    except:
        if os.path.exists(json_path):
            os.remove(json_path)
        exit(1)
