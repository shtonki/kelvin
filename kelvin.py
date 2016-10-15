from gen import *
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

PREDICT = True;
DAYSOFDATA = 1258;
dataset = [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
                ('vix',  CLOSE),
                ('xau',  CLOSE),                
            ];

def predictFor(clf, day):
    teX = getdata(dataset, day-10, day, noMemes=True);
    prd = prn.NNOut(teX, clf);
    o = rescale(teX[3][9]);
    p = rescale(prd[-1:][0]);
    return o, p;

if (__name__ == "__main__"):
    TRIALS = 1;
    for i in range(TRIALS):
        CO = 1257 if PREDICT else 1000;
        clf = prn.CreateNN([len(dataset),15,1],dIn=[0],dIntern=[1,2],dOut=[])

        trX, trY = getdata(dataset, 0, CO);

        clf = prn.train_LM(trX, trY, clf,verbose=True,k_max=20,E_stop=0.2)

        if (PREDICT):
            o, p = predictFor(clf, 1258);
            print(o, p);
        else:
            cr, inc, profit = 0, 0, 0;
            for i in range(CO, 1257):
                o, p = predictFor(clf, i)
                a = getCloseFor(i)
                if (p*0.99 > o):
                    profit += a*0.995 - o;
                    if (a > o):
                        cr += 1;
                    else:
                        incr += 1;
            print(cr, inc);
            print(profit);
