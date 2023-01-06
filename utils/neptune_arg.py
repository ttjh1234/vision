import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from collections import defaultdict as ddict


def fetch_neptune_key(path):
    key_arg=ddict(str)
    try:
        f = open(path,'r')
        while True:
            line=f.readline()
            if not line: 
                break
            line=line[:-1]
            key,value=str.split(line,':')
            key_arg[key]=value
        f.close
    except:
        print("There is no such file") 
    finally:        
        return key_arg
