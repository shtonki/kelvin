from gen import *
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

PREDICT = False;
DAYSOFDATA = 1258;
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
    teX = getdata(dataset, day-10, day, noMemes=True);
    prd = prn.NNOut(teX, clf);
    o = rescale(teX[3][9]);
    p = rescale(prd[-1:][0]);
    return o, p;

if (__name__ == "__main__"):
    pss = [];
    TRIALS = 10;
    for i in range(TRIALS):
        CO = 1257 if PREDICT else 1100;
        clf = prn.CreateNN([len(dataset),10,1],dIn=[0],dIntern=[1,2],dOut=[])

        trX, trY = getdata(dataset, 0, CO);

        clf = prn.train_LM(trX, trY, clf,verbose=True,k_max=10,E_stop=0.1)

        xd = [(predictFor(clf, i), getCloseFor(i)) for i in range(1207, 1257)];

        CORRECTION = -sum(map(lambda x: x[0][1] - x[1], xd))/50;
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
            for i in range(CO, CO+100):
                o, p = predictFor(clf, i);
                a = getCloseFor(i);
                p = p + CORRECTION;
                os.append(o);
                ps.append(p);
                az.append(a);
                a = a*0.9995;
                #   print(o, p, a);
                if (p > o):
                    profit += a - o;
                    if (a > o):
                        cr += 1;
                    else:
                        inc += 1;
            print(cr, inc);
            print(profit);
            #plt.plot(os);
            plt.plot(ps);
            plt.plot(az);
            plt.show();

    pss.sort();
    print(pss);
