from gen import getdata, OPEN, HIGH, LOW, CLOSE
import matplotlib.pyplot as plt
import numpy as np
import pyrenn as prn

if (__name__ == "__main__"):
    TRIALS = 1;
    for i in range(TRIALS):

        clf = prn.CreateNN([4, 16, 1]);
        X, y = getdata(
            [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
            ],
            0, 
            1000);
        clf = prn.train_LM(np.array(X), np.array(y), clf, verbose=True, k_max=50);

        X, y = getdata(
            [
                ('gspc', OPEN),
                ('gspc', HIGH),
                ('gspc', LOW),
                ('gspc', CLOSE),
            ],
            1001, 
            1200);

        p = prn.NNOut(X, clf);
        plt.plot(p);
        plt.plot(y);
        plt.show();