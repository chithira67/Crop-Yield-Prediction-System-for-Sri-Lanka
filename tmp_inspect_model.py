import joblib
import numpy as np
m = joblib.load('model/best_model.pkl')
print(type(m))
# create two sample feature vectors of length matching m.n_features_in_ if available
try:
    n = m.n_features_in_
    print('n_features_in_', n)
    x1 = np.zeros((1,n))
    x2 = np.ones((1,n))
    print('pred1', m.predict(x1))
    print('pred2', m.predict(x2))
except Exception as e:
    print('error', e)
