import pandas as pd
import numpy as np


def cf_user_based(user_id, interact, similarity="cosine"):

        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.stats import pearsonr


        """
        Compute rating predictions 
        ---
        Arg
        - interact: DataFrame
        - similarity: "cosine", "pearson"
        Ret

        """

        assert similarity in ["cosine", "pearson"], "Similarity should be 'cosine' or 'pearson'."

        sim = 0
        rat = np.zeros(len(interact[:, 0]))
        tar = interact[user_id, :]

        for user_index, user in enumerate(interact):

                if user_index == user_id:
                        continue
                
                s = 0
                if similarity == "cosine":
                        s = cosine_similarity(tar.reshape(1, -1), user.reshape(1, -1))[0, 0]
                else:
                        s = pearsonr(tar, user)[0]
                
                sim += s
                rat = [ u_r*s + r for index, (u_r, r) in enumerate(zip(user, rat))]
                
        rat = rat / s
        return rat




class MF():

        def __init__(self, R, K, alpha, beta, iterations):
                """
                Perform matrix factorization to predict empty
                entries in a matrix.

                Arguments
                - R (ndarray)   : user-item rating matrix
                - K (int)       : number of latent dimensions
                - alpha (float) : learning rate
                - beta (float)  : regularization parameter
                """

                self.R = R
                self.num_users, self.num_items = R.shape
                self.K = K
                self.alpha = alpha
                self.beta = beta
                self.iterations = iterations

        def train(self):
                # Initialize user and item latent feature matrice
                self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
                self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

                # Initialize the biases
                self.b_u = np.zeros(self.num_users)
                self.b_i = np.zeros(self.num_items)
                self.b = np.mean(self.R[np.where(self.R != 0)])

                # Create a list of training samples
                self.samples = [
                        (i, j, self.R[i, j])
                        for i in range(self.num_users)
                        for j in range(self.num_items)
                        if self.R[i, j] > 0
                ]

                # Perform stochastic gradient descent for number of iterations
                training_process = []
                for i in range(self.iterations):
                        np.random.shuffle(self.samples)
                        self.sgd()
                        mse = self.mse()
                        training_process.append((i, mse))
                        if (i+1) % 10 == 0:
                                print("Iteration: %d ; error = %.4f" % (i+1, mse))

                return training_process

        def mse(self):
                """
                A function to compute the total mean square error
                """
                xs, ys = self.R.nonzero()
                predicted = self.full_matrix()
                error = 0
                for x, y in zip(xs, ys):
                        error += pow(self.R[x, y] - predicted[x, y], 2)
                return np.sqrt(error)

        def sgd(self):
                """
                Perform stochastic graident descent
                """
                for i, j, r in self.samples:
                        # Computer prediction and error
                        prediction = self.get_rating(i, j)
                        e = (r - prediction)

                # Update biases
                self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
                self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

                # Update user and item latent feature matrices
                self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
                self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

        def get_rating(self, i, j):
                """
                Get the predicted rating of user i and item j
                """
                prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
                return prediction

        def full_matrix(self):
                """
                Computer the full matrix using the resultant biases, P and Q
                """
                return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
                        
                        


