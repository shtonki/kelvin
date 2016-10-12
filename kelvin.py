from lstm import lstm
from gen import gendata

clf = lstm(3, 1);

for i in range(1000):
    o, X, y = gendata(0, 7);
    clf.train(X, y);

o, X, y = gendata(1000, 7);
print(clf.predict(X), y);