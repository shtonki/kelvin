from lstm import lstm
from gen import gendata
import matplotlib.pyplot as plt
from numpy import mean

if (__name__ == "__main__"):
    _, X, _ = gendata(0, 3);
    INPUTDIM = len(X[0]);

    DAYS = 6;
    TRIALS = 10;
    BIGASKQUOTA = 0.99976512037;
    BUYTHRESHHOLD = 0.985;

    ess = [];
    buckets = [[0, 0], [0, 0]];
    for i in range(TRIALS):

        clf = lstm(INPUTDIM, 1, 0.85);

        for i in range(1000):
            o, X, y = gendata(i, DAYS);
            clf.train(X, y);

        profit = 0;
        for i in range(1000, 1230):
            o, X, y = gendata(i, DAYS);
            p = 2*o*clf.predict(X)[0][0];
            c = X[-1:][0][3]*2*o;
            a = 2*o*y[-1:][0][0];

            predBuy = 1 if p*BUYTHRESHHOLD > c else 0;
            actualProfit = 1 if a*BIGASKQUOTA - c > 0 else 0;

            buckets[predBuy][actualProfit] += 1;

            if (predBuy == 1):
                profit += a*BIGASKQUOTA - c;

        ess.append(profit);
    print("DAYS...........", DAYS);
    print("TRIALS.........", TRIALS);
    print("BIGASKQUOTA....", BIGASKQUOTA);
    print("BUYTHRESHHOLD..", BUYTHRESHHOLD);
    print("Mean profit....", mean(ess))
    print(buckets[0], "    ", buckets[0][0]/(buckets[0][0] + buckets[0][1]));
    print(buckets[1], "    ", buckets[1][1]/(buckets[1][0] + buckets[1][1]));

    #plt.show();