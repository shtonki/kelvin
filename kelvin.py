from gen import getdata, OPEN, HIGH, LOW, CLOSE
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

if (__name__ == "__main__"):
    TRIALS = 1;
    for i in range(TRIALS):

        clf = prn.CreateNN([4, 16, 1], dIn=[1,2],dIntern=[],dOut=[1,2,3]);
        X, y = getdata(
            [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
            ],
            0, 
            800);
        clf = prn.train_LM(X, y, clf, verbose=True, k_max=10);

        X0, y0 = getdata(
            [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
            ],
            775, 
            800);

        X, y = getdata(
            [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
            ],
            800, 
            1200);

        p = prn.NNOut(X, clf, P0=X0, Y0=y0);
        #p = np.zeros_like(p);
        print(np.mean(list(map(lambda x, y: abs(x-y), p, y))));
        plt.plot(p);
        plt.plot(y);
        plt.show();