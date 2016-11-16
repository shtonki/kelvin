from gen import *
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

PREDICT = False;
DAYSOFDATA = 1258;
RAMP = 15;
dataset = [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
                ('vix',  CLOSE),
                ('xau',  CLOSE),                
            ];

CORRECTION = 0;

def predictFor(clf, day):
    teX = getdata(dataset, day-RAMP, day, noMemes=True);
    prd = prn.NNOut(teX, clf);
    o = rescale(teX[3][RAMP-1]);
    p = rescale(prd[-1:][0]);
    return o, p;

if (__name__ == "__main__"):
    pss = [];
    TRIALS = 5;
    for i in range(TRIALS):
        CO = 1257 if PREDICT else 1100;
        clf = prn.CreateNN([len(dataset),12,1],dIn=[1],dIntern=[],dOut=[])

        trX, trY = getdata(dataset, 0, CO);

        clf = prn.train_LM(trX, trY, clf,verbose=False,k_max=15,E_stop=0.1)

        xd = [(predictFor(clf, i), getCloseFor(i)) for i in range(1207, 1257)];

        CORRECTION = 0-sum(map(lambda x: x[0][1] - x[1], xd))/50;
        print(CORRECTION);
        if (PREDICT):
            o, p = predictFor(clf, 1258);
            print(o, p);
            pss.append(p+CORRECTION);
        else:
            cr, inc, profit = 0, 0, 0;
            os = [];
            ps = [];
            az = [];
            ds = [];
            for i in range(CO, CO+100):
                o, p = predictFor(clf, i);
                a = getCloseFor(i);
                p = p + CORRECTION;
                os.append(o);
                ps.append(p);
                az.append(a);
                a = a-0.5;
                ds.append(p-a);
                #   print(o, p, a);
                if (p > o):
                    profit += a - o;
                    if (a > o):
                        cr += 1;
                    else:
                        inc += 1;
            print(cr+inc/2, inc/2);
            print(profit*2);
            #plt.plot(os);
            #plt.plot(ps);
            #plt.plot(az);
            #plt.plot(ds);
            plt.show();

    pss.sort();
    print(pss);
