from lstm import lstm
from gen import gendata
import matplotlib.pyplot as plt
from numpy import mean

if (__name__ == "__main__"):
    DAYS = 6;
    TRIALS = 10;

    ess = [];
    cor = 0;
    inc = 0;
    for i in range(TRIALS):

        clf = lstm(6, 1, 0.8);

        for i in range(1000):
            o, X, y = gendata(0, DAYS);
            clf.train(X, y);

        profit = 0;
        for i in range(1000, 1230):
            o, X, y = gendata(i, DAYS);
            p = 2*o*clf.predict(X)[0][0];
            c = X[-1:][0][3]*2*o;
            a = 2*o*y[-1:][0][0];
            if (p > c):
                cor += 1 if a > c else 0;
                inc += 1 if a < c else 0;
                profit += a*0.99 - c;

        ess.append(profit);
    print(mean(ess))
    print(cor/(cor+inc));

    #plt.show();