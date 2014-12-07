import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def test(k):
    main = 2 * np.ones(k)
    sub = -1 * np.ones(k)
    super = -1 * np.ones(k)
    diags = [sub, main, super]
    coords = [-1, 0, 1]
    S = scipy.sparse.spdiags(diags, coords, k, k, format='csc')
    D = S.todense()
    start = time.time()
    for _ in xrange(1):
        scipy.sparse.linalg.inv(S)
    end = time.time()
    t1 = end - start
    # print 'Time for sparse: %.3f' % (end - start)
    start = time.time()
    for _ in xrange(1):
        np.linalg.inv(D)
    end = time.time()
    t2 = end - start
    return t1, t2
    # print 'Time for dense: %.3f' % (end - start)

if __name__ == '__main__':
    t1_vec = []
    t2_vec = []
    k_vec = np.linspace(10, 1000, 10)
    for k in k_vec:
        print k
        t1, t2 = test(k)
        t1_vec.append(t1)
        t2_vec.append(t2)
    plt.plot(k_vec, t1_vec, label='Sparse')
    plt.plot(k_vec, t2_vec, label='Dense')
    plt.legend()
    plt.show()
