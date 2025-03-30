/*Test evaluation multivariate Gaussian distribution.

import scipy
import numpy as np

x = 0.5*np.ones(6)
mean = np.ones(6)
cov = np.array([[1.0, 0.2, 0.0, 0.2, 0.3, 0.7],
                [0.2, 1.0, 0.0, 0.1, 0.7, 0.1],
                [0.0, 0.0, 1.0, 0.5, 0.1, 0.6],
                [0.2, 0.1, 0.5, 1.0, 0.4, 0.2],
                [0.3, 0.7, 0.1, 0.4, 1.0, 0.1],
                [0.7, 0.1, 0.6, 0.2, 0.1, 1.0]])

norm = scipy.stats.multivariate_normal(mean=mean, cov=cov)

norm.pdf(x)*/

void test_gauss() {

    double x[6] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    double w[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    double S[6*6] = {1.0, 0.2, 0.0, 0.2, 0.3, 0.7,
                     0.2, 1.0, 0.0, 0.1, 0.7, 0.1,
                     0.0, 0.0, 1.0, 0.5, 0.1, 0.6,
                     0.2, 0.1, 0.5, 1.0, 0.4, 0.2,
                     0.3, 0.7, 0.1, 0.4, 1.0, 0.1,
                     0.7, 0.1, 0.6, 0.2, 0.1, 1.0};

    double g = multivariate_normal(x, w, S);
    printf("test_gauss: %+0.15E\n", g/0.021343284039406184);
}
