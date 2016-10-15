from gen import getdata, rescale, OPEN, HIGH, LOW, CLOSE
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn


CO, RU = 900, 10
DAYSOFDATA = 1258;
dataset = [
                ('gspc', OPEN),
                ('vix',  OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
                ('xau',  OPEN),                
            ];
if (__name__ == "__main__"):
    TRIALS = 1;
    for i in range(TRIALS):
        clf = prn.CreateNN([len(dataset),10,1],dIn=[0],dIntern=[1,2],dOut=[])

        trX, trY = getdata(dataset, 0, CO);

        clf = prn.train_LM(trX, trY, clf,verbose=True,k_max=20,E_stop=0.0001)

        teX, tey = getdata(dataset, CO, 1257);
        prd = prn.NNOut(teX, clf);
        plt.plot(prd);
        plt.plot(tey);
        #plt.show();
        a = [];
        cr, incr = 0, 0;
        profit = 0;
        for i in range(RU,len(tey)):
            o = rescale(teX[0][i]);
            p = rescale(prd[i]);
            a = rescale(tey[i]);
            if (p*0.98 > o):
                profit += a*0.999 - o;
                if (a > o):
                    cr += 1;
                else:
                    incr += 1;
        print(cr, incr);
        print(profit);
