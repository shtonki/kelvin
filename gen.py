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

def gendata(i, len):
    opn = dataSP[i][0];
    sp = normalize(dataSP[i:i+len]);
    aux = normalize(dataAUX[i:i+len]);
    vix = normalize(dataVIX[i:i+len]);
    rtnX = []
    rtnY = []
    for r in range(1, len-1):
        rtnX.append([
            sp[r][0],   
            sp[r][1],   
            sp[r][2],   
            sp[r][3],   
            aux[r][0],  
            vix[r][0],  
            ]);
        rtnY.append([
        sp[r+1][3], #next day's close
        #1 if sp[r+1][0] > sp[r+1][3] else 0
        ]); 
    return opn, rtnX, rtnY;

startDates = ['', '', '']
startDates[0],  dataSP = loadit("sp");
startDates[1], dataAUX = loadit("gold");
startDates[1], dataVIX = loadit("vix");

for i in range(1, len(startDates)):
    if (startDates[0] != startDates[i]):
        raise ValueError('Inconsistent start dates in data');

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
