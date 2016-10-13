import re
import pickle
import sys

def loadit(s):
    with open("data/" + s + ".p", "rb") as f:
        return pickle.load(f);

def normalize(sp):
    xd = sp[0][0];
    for r in range(len(sp)):
        sp[r] = list(map(lambda x: x/(2*xd), sp[r]));
    return sp;

def gendata(i, days):
    opn = tables['gspc'][i][OP];
    data = dict(map(lambda x: (x, normalize(tables[x][i:i+days])), sources))
    rtnX = []
    rtnY = []
    for r in range(1, days-1):
        rtnX.append([
            data['gspc'][r][OP],
            data['gspc'][r][HI],
            data['gspc'][r][LO],
            data['gspc'][r][CL],
            data['xau'][r][OP],
            data['vix'][r][OP],
            ]);
        rtnY.append([
        data['gspc'][r+1][3], #next day's close
        ]); 
    return opn, rtnX, rtnY;

OP = 0;
HI = 1;
LO = 2;
CL = 3;

sources = ['gspc', 'xau', 'vix'];
startDates, ds = list(zip(*list(map(lambda x: loadit(x), sources))));
tables = {}


for i in range(0, len(sources)):
    if (startDates[0] != startDates[i]):
        raise ValueError('Inconsistent start dates in data');
    tables[sources[i]] = ds[i];

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
