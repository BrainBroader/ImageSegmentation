import numpy as np


class GaussianMixtureModel:

    def __init__(self, K, max_iter=50, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.mean = None
        self.gamma = None

    def fit(self, X):
        """
        Calculates means and variances of the gaussian's using the EM algorithm.

        :param X: input data
        :return: A clustering model trained on the input data
        """
        N, M, D = X.shape

        # random initialization of mean, covariance and prior possibility
        self.mean, variance = np.random.rand(self.K, D), np.random.rand(self.K)
        prior = np.random.rand(self.K)
        temp = prior.sum()
        for i in range(self.K):
            prior[i] = prior[i] / temp

        # initialization to zero of gamma(zk) table and cost
        self.gamma = np.zeros([N, M, self.K])
        prev_cost = np.inf
        costs = []

        for iteration in range(self.max_iter):
            # E-step
            for i in range(N):
                for j in range(M):

                    # calculate numerator of gamma(zk) : prior[k] * N(x /mean, covariance)
                    numerator = np.ones(self.K)
                    for k in range(self.K):
                        numerator[k] = prior[k]*self.gaussian(self.mean[k, :], variance[k], X[i, j, :], D)
                    for k in range(self.K):
                        self.gamma[i, j, k] = numerator[k] / np.sum(numerator)

            # M-step
            for k in range(self.K):
                gamma_sum = np.sum(self.gamma[:, :, k])

                # calculate new prior possibilities
                prior[k] = gamma_sum / (N*M)

                # calculate new mean values
                for d in range(D):
                    self.mean[k, d] = np.sum(self.gamma[:, :, k] * X[:, :, d]) / gamma_sum

                # calculate new variance values
                for i in range(N):
                    for j in range(M):
                        for d in range(D):
                            variance[k] += (self.gamma[i, j, k] * np.square(X[i, j, d] - self.mean[k, d]))
                variance[k] = variance[k] / (D * gamma_sum)

            # Calculate cost
            new_cost = self.cost_function(N, M, prior, variance, X, D)
            costs.append(new_cost)
            print("Iteration " + str(iteration) + " : " + str(new_cost))

            if abs(new_cost - prev_cost) < self.tol:
                break
            prev_cost = new_cost

    @staticmethod
    def gaussian(mean, variance, x, D):

        gd = 1
        for i in range(D):
            gd *= np.exp(-(x[i] - mean[i]) ** 2 / (2*variance)) / np.sqrt(2 * np.pi * variance)

        return gd

    def cost_function(self, N, M, prior, variance, X, D):
        """
        Calculates the cost of the gaussian mixture model p(x) = sum(log(sum(pk * N(xn/μκ, Σκ))))
        :param N:  1st dimension of input data
        :param M: 2nd dimension of input data
        :param prior: prior possibility of each category
        :param variance: value of the variance of each gaussian
        :param X: input data
        :param D: 3rd dimension of input data
        :return: the cost
        """
        cost = 0
        for i in range(N):
            for j in range(M):
                temp = 0
                for k in range(self.K):
                    temp += prior[k] * self.gaussian(self.mean[k, :], variance[k], X[i, j, :], D)
                cost += np.log(temp)

        return cost

    def return_segmented_image(self, X):
        """
        Gets as input image data and returns the same image segmented in k categories according to trained model.
        :param X: image data
        :return: segmented image and reconstruction error
        """
        N, M, D = X.shape
        segmented = np.zeros_like(X)
        error = 0
        if self.mean is not None:

            for i in range(N):
                for j in range(M):
                    k = np.argmax(self.gamma[i, j, :])
                    for d in range(D):
                        segmented[i, j, d] = self.mean[k, d]
                        error += np.square(X[i, j, d] - self.mean[k, d])
            error = np.sqrt(error) / (N * M * D)
        return segmented, error
