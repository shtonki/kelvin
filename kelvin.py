from gen import *
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

PREDICT = False;
DAYSOFDATA = 1258;
RAMPDAYS = 5;
dataset = [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
          #      ('vix',  CLOSE),
          #      ('xau',  CLOSE),            
                ('TNX',  CLOSE),      
                ('N225',  CLOSE),          
            ];
dataset2 = [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
       #         ('vix',  CLOSE),
      #          ('xau',  CLOSE),            
       #         ('TNX',  CLOSE),      
       #         ('N225',  CLOSE),          
            ];
dataset3 = [
                ('gspc', OPEN),
      #          ('gspc', HIGH),
     #           ('gspc', LOW),
                ('gspc', CLOSE),
                ('vix',  CLOSE),
         #       ('xau',  CLOSE),            
                ('TNX',  CLOSE),      
          #      ('N225',  CLOSE),          
            ];



def predictFor(clf, day):
    teX = getdata(dataset, day-RAMPDAYS, day, noMemes=True);
    prd = prn.NNOut(teX, clf);
    o = rescale(teX[3][RAMPDAYS-1]);
    p = rescale(prd[-1:][0]);
    return o, p;

def predictFor2(clf2, day):
    teX2 = getdata(dataset2, day-RAMPDAYS, day, noMemes=True);
    prd2 = prn.NNOut(teX2, clf2);
    o = rescale(teX2[3][RAMPDAYS-1]);
    p2 = rescale(prd2[-1:][0]);
    return o, p2;

def predictFor3(clf3, day):
    teX3 = getdata(dataset3, day-RAMPDAYS, day, noMemes=True);
    prd3 = prn.NNOut(teX3, clf3);
    o = rescale(teX3[3][RAMPDAYS-1]);
    p3 = rescale(prd3[-1:][0]);
    return o, p3;




if (__name__ == "__main__"):
    pss = [];
    TRIALS = 1;
    for i in range(TRIALS):
        CO = 1257 if PREDICT else 1000;
        clf = prn.CreateNN([len(dataset),10,1],dIn=[1],dIntern=[],dOut=[])
        clf2 = prn.CreateNN([len(dataset2),10,1,1],dIn=[1],dIntern=[],dOut=[])
        clf3 = prn.CreateNN([len(dataset3),20,1],dIn=[1],dIntern=[],dOut=[])


        trX, trY = getdata(dataset, 0, CO);
        trX2, trY2 = getdata(dataset2, 0, CO);
        trX3, trY3 = getdata(dataset3, 0, CO);

        clf = prn.train_LM(trX, trY, clf,verbose=True,k_max=15,E_stop=0.1)
        clf2 = prn.train_LM(trX2, trY2, clf2,verbose=True,k_max=15,E_stop=0.1)
        clf3 = prn.train_LM(trX3, trY3, clf3,verbose=True,k_max=15,E_stop=0.1)


        if (PREDICT):
            o, p = predictFor(clf, 1258);
            o, p2 = predictFor2(clf2, 1258);
            o, p3 = predictFor3(clf3, 1258);
            x = (p + p2 + p3) / 3 
            print(o, x);


        else:
            cr, inc, cr2, inc2, profit = 0, 0, 0, 0, 0;
            os = [];
            ps = [];
            az = [];
            ds = [];
            for i in range(CO, CO+258):
                o, p = predictFor(clf, i);
                o2, p2 = predictFor2(clf2, i);
                o3, p3 = predictFor3(clf3, i);
                a = getCloseFor(i);
                os.append(o);
                ps.append(p);
                az.append(a);
                a = a-0.5;
                ds.append(p-a);
                #   print(o, p, a);

                if (p > o and p2 > o and p3 > o):
                    profit += a - o;
                    if (a > o):
                        cr += 1;
                    else:
                        inc += 1;

                if (p < o and p2 <o and p3 < o):
                    profit += o - a;
                    if (a < o):
                        cr2 += 1;
                    else:
                        inc2 += 1;


            print(cr, inc);
            print(cr2, inc2);
            print(profit);
            plt.plot(os);
            plt.plot(ps);
            plt.plot(az);
            #plt.plot(ds);
            plt.show();

    pss.sort();
    print(pss);
