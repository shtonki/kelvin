import re
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sin
from random import shuffle

def loadit(s):
    with open("data/" + s + ".p", "rb") as f:
        return pickle.load(f);

def getCloseFor(day):
    return rescale(tables['gspc'][1][3][day]);

def getdata(sources, startDay, endDay, noMemes = False):
    X = [];
    y = [];
    for (src, col) in sources:
        o, tbl = tables[src];
        X.append(tbl[col][startDay:endDay]);
    tmp = tables['gspc'][1][CLOSE];
    if (noMemes):
        return np.array(X);
    for d in range(startDay, endDay):
        y.append(tmp[d+1]);
    return np.array(X), np.array(y);

def rescale(v):
    return rsov*((v*(rsmx - rsmn))+rsmn)

rsmx, rsmn, rsov = 0, 0, 0;

OPEN  = 0;
HIGH  = 1;
LOW   = 2;
CLOSE = 3;

sources = ['gspc', 'xau', 'vix'];
startDates, ds = list(zip(*list(map(lambda x: loadit(x), sources))));
tables = {}


for i in range(0, len(sources)):
    if (startDates[0] != startDates[i]):
        raise ValueError('Inconsistent start dates in data');
    fk = np.array(ds[i]).T;
    
    ov = fk[OPEN][0];
    for rw in range(0, 4):
        fk[rw] = list(map(lambda x: x/ov, fk[rw]));
    mx = fk.max();
    mn = fk.min();
    kolonp = mx-mn;
    xd = [0]*4
    for j in range(0, 4):
        xd[j] = list(map(lambda x: (x-mn)/kolonp, fk[j]));
    tables[sources[i]] = ov, xd;
    if (sources[i] == 'gspc'):
        rsmx, rsmn, rsov = mx, mn, ov;


if (__name__ == "__main__" and len(sys.argv) == 2):
    fh = sys.argv[1];
    print("Converting", fh + ".csv");
    with open(fh + ".csv", "r") as f:
        con = f.readlines()[1:];
    con.reverse();
    startDate = con[0][:10];
    rex = '(\d\d\d\d-\d\d-\d\d),([\d\.]*),([\d\.]*),([\d\.]*),([\d\.]*),([\d\.]*),([\d\.]*)\\n'
    d = []
    for c in con:
        mt = re.match(rex, c);
        d.append([
            #mt.group(1), 
            float(mt.group(2)), 
            float(mt.group(3)), 
            float(mt.group(4)), 
            float(mt.group(5)),
                ]);
    with open(fh + ".p", "wb") as f:
        pickle.dump((startDate, d), f);
