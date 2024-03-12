# This script downloads the files of $REWIND_N_MONTHS last months of wikipedia article consultation starting from $TIME_END.
# All this data is then stored in a Mongo Database.
# We should at the end of this process have a database "wikipedia_month_consultation" that will contain the monthly consultation of 
# every english wikipedia article.

from multiprocessing import pool
import random
import subprocess
import sys
import threading
import requests
from copy import deepcopy
import time
import os
from pymongo import MongoClient
import re
import bz2
from build.config import *
import argparse
import indexed_bzip2
import os.path as osp
from build.config import REWIND_N_MONTHS


# WIKI_PAGEVIEWS_DUMPS_URL = "https://dumps.wikimedia.org/other/pageview_complete/monthly/"

def main(version : str, n_rewind : int):
    REWIND_N_MONTHS = n_rewind
    WIKIPEDIA_DATE = OLD_WIKIDATA_DATE if version == 'old' else NEW_WIKIDATA_DATE
    WIKIPEDIA_DATE = WIKIPEDIA_DATE[:6]
    class ProcessState:
        NOT_FROM_WIKI = 1,
        FROM_WIKI_BUT_BAD_ID = 2,
        OK = 3

    CHUNK_SIZE = 100*2**20
    WIKIPEDIA_MONTH_CONSULT_MDB_COLLECTION_NAME = "%s_wikipedia_month_consultation" % version
    WIKIPEDIA_TO_FETCH = b'en.wikipedia'
    # TIME_END : Last month to study, Format = (MONTH, YEAR)
    # REWIND_N_MONTHS : How many months to go back in time
    TIME_END = int(WIKIPEDIA_DATE[4:6])-1, int(WIKIPEDIA_DATE[:4])
    if TIME_END[0] == 0:
        TIME_END = 12, TIME_END[1]-1
    TEMPORARY_DIRECTORY = os.path.join(STORAGE_FOLDER, "pageviews_tmp")
    PATTERN_FILE = re.compile(r'pageviews-([0-9]{4})([0-9]{2})-user\.bz2')
    print('Drop existing database in MongoDB...', end=' ')
    client = MongoClient(MONGO_URL)
    db = client[MONGODB_NAME]
    collection_tmp = db[WIKIPEDIA_MONTH_CONSULT_MDB_COLLECTION_NAME]
    collection_tmp.drop()
    print('OK.')

    class TimeItContextManager:
        def __init__(self, name : str) -> None:
            self.name = name

        def __enter__(self):
            print(f'{self.name}', end=' ')
            self.time = time.time()

        def __exit__(self, exc_type, exc_value, exc_tb):
            self.execution_time = time.time() - self.time
            print(f': {self.execution_time} sec')

    timeit_read_chunk = TimeItContextManager('Read chunk')
    timeit_continue_line = TimeItContextManager('Continue to end of line')
    timeit_build_queries = TimeItContextManager('Building update queries')
    timeit_mongo_insert = TimeItContextManager('MongoDB update chunk')

    def get_url(month : int, year : int, with_file=False):
        """Get URL of monthly pageviews dataset given year and month

        Args:
            month (int, optional): Month
            year (int): Year
            with_file (bool, optional): If True, get the download URL of the file of interest. Defaults to False.

        Returns:
            URL of the pageviews dataset corresponding to the given month and year
        """
        if month is not None and month < 10:
            month = '0' + str(month)
        url = f"https://dumps.wikimedia.org/other/pageview_complete/monthly/{year}/"
        if month is not None:
            url += f"{year}-{month}/"
            if with_file:
                url += f"pageviews-{year}{month}-user.bz2"
        return url

    def check_url(url : str):
        response = requests.get(url)
        return response.status_code == 200

    def get_months(end_time, n_rewind):
        end_time = list(end_time)
        res = [deepcopy(end_time)]
        while n_rewind > 1:
            end_time[0] -= 1
            if end_time[0] == 0:
                end_time[0] = 12
                end_time[1] -= 1
            res.append(deepcopy(end_time))
            n_rewind -= 1
        return res


    class BzipFile(bz2.BZ2File):
        def __init__(self, path : str) -> None:
            super().__init__(path, mode='rb')
            self.path = path
            self.buffer = b''
            self.forget = False

        def readline(self, size = -1) -> bytes:
            if self.forget:
                b = self.buffer
                self.buffer = b''
                self.forget = False
                return b
            self.buffer = super().readline(size)
            return self.buffer
        
        def read(self, size: int | None = -1) -> bytes:
            if not self.forget:
                return super().read(size)
            size_ = max(0,size-len(self.buffer))
            res = self.buffer[:size] + super().read(size_)
            self.buffer = self.buffer[size_:]
            if len(self.buffer) == 0:
                self.forget = False
            return res
        
        def forgetline(self):
            self.forget = True
        
    list_dates = get_months(TIME_END, REWIND_N_MONTHS)
    urls_to_download = []
    remaining_dates = []
    os.makedirs(TEMPORARY_DIRECTORY, exist_ok=True)
    print('Collecting URLs...', end=' ')
    for d in list_dates:
        if not check_url(get_url(*d)):
            print(f'WARNING : pageviews is not available for {d[0]}-{d[1]}. Skipping this month !')
        else:
            url = get_url(*d, with_file=True)
            urls_to_download.append(url)
            remaining_dates.append(d)
    print('OK.')

    # Download dumps
    commands = [('bash %s -c ' %  osp.join(osp.dirname(__file__), '../utils/wget_lock.sh')) + x for x in urls_to_download ]
    # Mitigate eventual collisions if downloading --version new and --version old at the same time (Clearly not the best solution to this) 
    random.shuffle(commands)

    def target(command : str):
        while True:
            state = subprocess.run(('cd %s && ' % TEMPORARY_DIRECTORY) + command, shell=True, stdin=None, stdout=sys.stdout, stderr=sys.stderr)
            if state.returncode == 0:
                break
            time.sleep(10)
    p = pool.ThreadPool(2)
    p.map(target, commands)

    
    

    list_files = os.listdir(TEMPORARY_DIRECTORY)
    date2file = {}
    date2size = {}
    date2last_ckp = {}
    date2last = {}
    total_size = 0
    dates_to_process = [str(x[1]) + str(x[0]).zfill(2) for x in remaining_dates]
    for file in list_files:
        if not (match := re.match(r'^pageviews-([0-9]{6})-user\.bz2$', file)):
            continue
        if match.group(1) not in dates_to_process:
            continue
        year, month = re.match(PATTERN_FILE, file).groups()
        year_month = year+month
        date2file[year_month] = indexed_bzip2.open(os.path.join(TEMPORARY_DIRECTORY, file), parallelization=8)
        # date2file[year_month].seek(6*CHUNK_SIZE)
        date2size[year_month] = os.path.getsize(os.path.join(TEMPORARY_DIRECTORY, file))
        total_size += date2size[year_month]
        date2last_ckp[year_month] = 0
        
        # Last title, last_id
        date2last[year_month] = (None, None)


    def process_line(line : bytes, last_title : str = None, last_id : int = None):
        splits = line.split(b' ')
        if len(splits) != 6:
            ret = ProcessState.FROM_WIKI_BUT_BAD_ID if splits[0] == WIKIPEDIA_TO_FETCH else ProcessState.NOT_FROM_WIKI
            print('WARNING : Line skipped = %s' % splits)
            return ret, splits
        splits[-2] = int(splits[-2])
        project, title, page_id, _, month_consultation = splits[:-1]
        if splits[0] != WIKIPEDIA_TO_FETCH:
            return ProcessState.NOT_FROM_WIKI, (project, title, page_id, month_consultation)
        if page_id == b'null' and title == last_title:
            page_id = last_id
        if page_id == b'null':
            return ProcessState.FROM_WIKI_BUT_BAD_ID, (project, title, page_id, month_consultation)
        return ProcessState.OK, (project, title, page_id, month_consultation)


    while True:
        t1 = time.time()
        with timeit_read_chunk:
            chunks = {k : v.read(CHUNK_SIZE//len(date2file)) + v.readline()[:-1] for k, v in date2file.items()}
        

        # Drop fininshed files
        to_remove = []
        for year_month, chunk in chunks.items():
            if len(chunk) == 0:
                to_remove.append(year_month)
        for r in to_remove:
            del chunks[r]
            date2file[r].close()
            del date2file[r]

        # Optimization trick : If current chunk do not instersect with en.wikipedia skip it, and if chunk is after en.wikipedia stop file
        to_remove = []
        for year_month in chunks.keys():
            chunk = chunks[year_month]
            first_line_domain = chunk[:chunk.find(b'\n')-1].split(b' ')[0]
            last_line_domain = chunk[chunk.rfind(b'\n')+1:].split(b' ')[0]
            if last_line_domain < WIKIPEDIA_TO_FETCH:
                chunks[year_month] = b''
            elif first_line_domain > WIKIPEDIA_TO_FETCH:
                to_remove.append(year_month)
        
        for r in to_remove:
            del chunks[r]
            date2file[r].close()
            del date2file[r]

        # If no more files to process ==> exit
        if len(date2file) == 0:
            break

        if all(len(c) == 0 for c in chunks.values()):
            print('Nothing in this chunk!\n')
            continue

        for year_month in chunks.keys():
            file = date2file[year_month]
            with TimeItContextManager('Process lines {}'.format(year_month)):
                if len(chunks[year_month]) == 0:
                    chunks[year_month] = []
                    continue
                chunks[year_month] = chunks[year_month].split(b'\n')

                last_title, last_id = None, None
                splits = []
                for l in chunks[year_month]:
                    success, res = process_line(l, last_title, last_id)
                    if success != ProcessState.OK:
                        continue
                    _, last_title, last_id, month_consultation = res

                    # Combine views of different platforms
                    if len(splits) and last_title == splits[-1][0]:
                        splits[-1][2] += month_consultation
                    else:
                        splits.append([last_title, last_id, month_consultation])
                date2last[year_month] = (last_title, last_id)
                chunks[year_month] = splits

            with TimeItContextManager('Continue until new page id {}'.format(year_month)):
                # Continue until new page id appears
                current_id = last_id
                current_title = last_title
                while True:
                    line = file.readline()[:-1]
                    success, res = process_line(line, current_title, current_id)
                    if success != ProcessState.OK:
                        break
                    _, current_title, current_id, month_consultation = res
                    if current_title != last_title:
                        file.seek(file.tell()-len(line)-1)
                        break
                    splits[-1][2] += month_consultation      
        
        # In case there is nothing in this chunk
        if all(x == (None, None) for x in date2last.values()):
            print('Nothing in this chunk!\n')
            continue

        # Find the furthest title in all files
        max_title = max([x for x in date2last.values() if x[0] is not None], key=lambda x:x[0])[0]

        # for year_month, chunk in chunks.items():
        #     with open('../logs/chunk_{}.log'.format(year_month), 'w') as f:
        #         data = [(x.decode(), y.decode(), z) for x,y,z in chunk]
        #         json.dump(data, f, indent=4)

        # Forward all files to this max_title
        for year_month in chunks.keys():
            with TimeItContextManager('Forward to max_title {}'.format(year_month)):
                continue_to_next = False
                file = date2file[year_month]
                splits = chunks[year_month]
                last_title, last_id = date2last[year_month]
                if last_title is None or last_title < max_title:
                    while True:
                        line = file.readline()[:-1]
                        success, res = process_line(line, last_title, last_id)
                        if success != ProcessState.OK:
                            if success == ProcessState.NOT_FROM_WIKI and res[0] > WIKIPEDIA_TO_FETCH:
                                break
                            continue
                        _, last_title, last_id, month_consultation = res

                        if len(splits) and last_title == splits[-1][0]:
                            splits[-1][2] += month_consultation
                        else:
                            splits.append([last_title, last_id, month_consultation])
                        
                        if last_title == max_title:
                            continue_to_next = True
                            break
                        elif last_title > max_title:
                            file.seek(file.tell()-len(line)-1)
                            splits.pop()
                            break
            
            if continue_to_next:
                # Continue until new page id appears
                current_id = last_id
                current_title = last_title
                while True:
                    line = file.readline()[:-1]
                    success, res = process_line(line, current_title, current_id)
                    if success != ProcessState.OK:
                        break
                    _, current_title, current_id, month_consultation = res
                    if current_title != last_title:
                        file.seek(file.tell()-len(line)-1)
                        break
                    splits[-1][2] += month_consultation
        
        # Now, we have all the files at position max_title
        # Build the month consultation per page dictionary
        inserts = {}
        with timeit_build_queries:
            for year_month in chunks.keys():
                for title, id_, month_consultation in chunks[year_month]:
                    title = title.decode().replace('_', ' ')
                    if title in inserts:
                        inserts[title]['month_consults'].append({'year_month' : year_month, 'n' : month_consultation})
                    else:
                        inserts[title] = {'_id' : title, 'page_id' : id_.decode(), 
                                                'month_consults' : [{'year_month' : year_month, 'n' : month_consultation}]}

        with timeit_mongo_insert:
            collection_tmp.insert_many(list(inserts.values()), ordered=False)

        t2 = time.time()
        processed_all = 0

        for year_month, file in date2file.items():
            pos = file.tell_compressed() // 8
            processed_file = pos - date2last_ckp[year_month]
            processed_all += processed_file
            date2last_ckp[year_month] = pos
            print('Progress File {} : {}/{}GB'.format(year_month, round(pos/2**30, 2), round(date2size[year_month]/2**30, 2)))
        print('\nTotal progress : {}/{}GB'.format(round(sum(date2last_ckp.values())/2**30, 2), round(total_size/2**30, 2)))
        print('Process speed : {}MB/s\n'.format(round(processed_all/(t2-t1)/2**20, 2)))
        print('Chunk finished!  {}sec'.format(t2-t1))
    print('The database creation is a success!')

if __name__ == '__main__':
    def check_positive(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        help="Wikidata version for which to collect views stats",
        choices=['old', 'new'],
        required=True,
    )
    # parser.add_argument(
    #     "--n_rewind",
    #     type=check_positive,
    #     help="How many months the script should retrieve. This number must be positive.",
    #     default=12
    # )
    args = parser.parse_args()
    t1 = time.time()
    main(
        args.version,
        REWIND_N_MONTHS
    )
    print('Execution time : %s' % (time.time() - t1))
    print('Finished!')
