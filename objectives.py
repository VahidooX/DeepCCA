import theano.tensor as T


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """
    def inner_cca_objective(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented by Theano tensor operations, and does not work on Tensorflow backend
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = y_pred.shape[1]//2

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = y_pred[:, 0:o1].T
        H2 = y_pred[:, o1:o1+o2].T

        m = H1.shape[1]

        H1bar = H1 - (1.0 / m) * T.dot(H1, T.ones([m, m]))
        H2bar = H2 - (1.0 / m) * T.dot(H2, T.ones([m, m]))

        SigmaHat12 = (1.0 / (m - 1)) * T.dot(H1bar, H2bar.T)
        SigmaHat11 = (1.0 / (m - 1)) * T.dot(H1bar, H1bar.T) + r1 * T.eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * T.dot(H2bar, H2bar.T) + r2 * T.eye(o2)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = T.nlinalg.eigh(SigmaHat11)
        [D2, V2] = T.nlinalg.eigh(SigmaHat22)

        # Added to increase stability
        posInd1 = T.gt(D1, eps).nonzero()[0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = T.gt(D2, eps).nonzero()[0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = T.dot(T.dot(V1, T.nlinalg.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = T.dot(T.dot(V2, T.nlinalg.diag(D2 ** -0.5)), V2.T)

        Tval = T.dot(T.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            # all singular values are used to calculate the correlation
            corr = T.sqrt(T.nlinalg.trace(T.dot(Tval.T, Tval)))
        else:
            # just the top outdim_size singular values are used
            [U, V] = T.nlinalg.eigh(T.dot(Tval.T, Tval))
            U = U[T.gt(U, eps).nonzero()[0]]
            U = U.sort()
            corr = T.sum(T.sqrt(U[0:outdim_size]))

        return -corr

    return inner_cca_objective

