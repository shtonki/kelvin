import re
import pickle
import sys

def loadit(s):
    with open("data/" + s + ".p", "rb") as f:
        return pickle.load(f);

def gendata(i):
    sp = dataSP[i:i+10];
    xd = sp[0];
    for r in range(10):
        sp[r] = list(map(lambda x, y: x/(y*2), sp[r], xd));
    return sp;

dataSP = loadit("sp");
dataAUX = loadit("gold");


if (__name__ == "__main__" and len(sys.argv) == 2):
    fh = sys.argv[1];
    print("Converting", fh + ".csv");
    with open(fh + ".csv", "r") as f:
        con = f.readlines()[1:];
    con.reverse();
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
        pickle.dump(d, f);
