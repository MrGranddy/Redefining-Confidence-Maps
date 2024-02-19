from typing import Literal, Tuple, Optional

import numpy as np
import cv2

from scipy.sparse import csc_matrix
from scipy.signal import hilbert

from scipy import sparse, ndimage as ndi

# executive summary for next code block: try to import umfpack from
# scipy, but make sure not to raise a fuss if it fails since it's only
# needed to speed up a few cases.
# See discussions at:
# https://groups.google.com/d/msg/scikit-image/FrM5IGP6wh4/1hp-FtVZmfcJ
# https://stackoverflow.com/questions/13977970/ignore-exceptions-printed-to-stderr-in-del/13977992?noredirect=1#comment28386412_13977992
try:
    from scipy.sparse.linalg.dsolve.linsolve import umfpack
    old_del = umfpack.UmfpackContext.__del__

    def new_del(self):
        try:
            old_del(self)
        except AttributeError:
            pass
    umfpack.UmfpackContext.__del__ = new_del
    UmfpackContext = umfpack.UmfpackContext()
except ImportError:
    UmfpackContext = None

try:
    from pyamg import ruge_stuben_solver
    amg_loaded = True
except ImportError:
    amg_loaded = False

from scipy.sparse.linalg import cg, spsolve

class UltrasoundConfidenceMap:
    """Compute confidence map from an ultrasound image.
    This transform uses the method introduced by Karamalis et al. in https://doi.org/10.1016/j.media.2012.07.005.
    It generates a confidence map by setting source and sink points in the image and computing the probability
    for random walks to reach the source for each pixel.

    Args:
        alpha (float, optional): Alpha parameter. Defaults to 2.0.
        beta (float, optional): Beta parameter. Defaults to 90.0.
        gamma (float, optional): Gamma parameter. Defaults to 0.05.
        mode (str, optional): 'RF' or 'B' mode data. Defaults to 'B'.
        sink_mode (str, optional): Sink mode. Defaults to 'all'.
        sink_mask (np.ndarray, optional): Sink mask. Defaults to None.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 90.0,
        gamma: float = 0.05,
        mode: Literal["RF", "B"] = "B",
        sink_mode: Literal["all", "mid", "min", "mask"] = "all",
        sink_mask: Optional[np.ndarray] = None,
        backend: Literal["scipy", "octave"] = "scipy",
    ):

        # The hyperparameters for confidence map estimation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.sink_mode = sink_mode
        self.sink_mask = sink_mask

        if self.sink_mode == "mask" and self.sink_mask is None:
            raise ValueError("Sink mask must be provided when sink mode is mask, please use 'sink_mask' argument.")

        # The precision to use for all computations
        self.eps = np.finfo("float64").eps

        if backend == "octave":
            from oct2py import Oct2Py
            # Octave instance for computing the confidence map
            self.oc = Oct2Py()

    def sub2ind(self, size: Tuple[int], rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Converts row and column subscripts into linear indices,
        basically the copy of the MATLAB function of the same name.
        https://www.mathworks.com/help/matlab/ref/sub2ind.html

        This function is Pythonic so the indices start at 0.

        Args:
            size Tuple[int]: Size of the matrix
            rows (np.ndarray): Row indices
            cols (np.ndarray): Column indices

        Returns:
            indices (np.ndarray): 1-D array of linear indices
        """
        indices = rows + cols * size[0]
        return indices

    def get_seed_and_labels(self, data : np.ndarray, sink_mode: str = "all", sink_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get the seed and label arrays for the max-flow algorithm

        Args:
            data: Input array
            sink_mode (str, optional): Sink mode. Defaults to 'all'.
            sink_mask (np.ndarray, optional): Sink mask. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Seed and label arrays
        """

        # Seeds and labels (boundary conditions)
        seeds = np.array([], dtype="float64")
        labels = np.array([], dtype="float64")

        # Indices for all columns
        sc = np.arange(data.shape[1], dtype="float64")

        # SOURCE ELEMENTS - 1st matrix row
        # Indices for 1st row, it will be broadcasted with sc
        sr_up = np.array([0])
        seed = self.sub2ind(data.shape, sr_up, sc).astype("float64")
        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 1
        label = np.ones_like(seed)
        labels = np.concatenate((labels, label))

        # SINK ELEMENTS - last image row
        if sink_mode == "all":
            sr_down = np.ones_like(sc) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc).astype("float64")       
        elif sink_mode == "mid":
            sc_down = np.array([data.shape[1] // 2])
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")
        elif sink_mode == "min":
            # Find the minimum value in the last row
            min_val = np.min(data[-1, :])
            min_idxs = np.where(data[-1, :] == min_val)[0]
            sc_down = min_idxs
            sr_down = np.ones_like(sc_down) * (data.shape[0] - 1)
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")
        elif sink_mode == "mask":
            coords = np.where(sink_mask != 0)
            sr_down = coords[0]
            sc_down = coords[1]
            seed = self.sub2ind(data.shape, sr_down, sc_down).astype("float64")

        seed = np.unique(seed)
        seeds = np.concatenate((seeds, seed))

        # Label 2
        label = np.ones_like(seed) * 2
        labels = np.concatenate((labels, label))

        return seeds, labels

    def normalize(self, inp: np.ndarray) -> np.ndarray:
        """Normalize an array to [0, 1]"""
        return (inp - np.min(inp)) / (np.ptp(inp) + self.eps)

    def attenuation_weighting(self, A: np.ndarray, alpha: float) -> np.ndarray:
        """Compute attenuation weighting

        Args:
            A (np.ndarray): Image
            alpha: Attenuation coefficient (see publication)

        Returns:
            W (np.ndarray): Weighting expressing depth-dependent attenuation
        """

        # Create depth vector and repeat it for each column
        Dw = np.linspace(0, 1, A.shape[0], dtype="float64")
        Dw = np.tile(Dw.reshape(-1, 1), (1, A.shape[1]))

        W = 1.0 - np.exp(-alpha * Dw)  # Compute exp inline

        return W

    def confidence_laplacian(
        self, P: np.ndarray, A: np.ndarray, beta: float, gamma: float
    ) -> csc_matrix:
        """Compute 6-Connected Laplacian for confidence estimation problem

        Args:
            P (np.ndarray): The index matrix of the image with boundary padding.
            A (np.ndarray): The padded image.
            beta (float): Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma (float): Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            L (csc_matrix): The 6-connected Laplacian matrix used for confidence map estimation.
        """

        m, _ = P.shape

        P = P.T.flatten()
        A = A.T.flatten()

        p = np.where(P > 0)[0]

        i = P[p] - 1  # Index vector
        j = P[p] - 1  # Index vector
        # Entries vector, initially for diagonal
        s = np.zeros_like(p, dtype="float64")

        vl = 0  # Vertical edges length

        edge_templates = [
            -1,  # Vertical edges
            1,
            m - 1,  # Diagonal edges
            m + 1,
            -m - 1,
            -m + 1,
            m,  # Horizontal edges
            -m,
        ]

        vertical_end = None
        diagonal_end = None

        for iter_idx, k in enumerate(edge_templates):

            Q = P[p + k]

            q = np.where(Q > 0)[0]

            ii = P[p[q]] - 1
            i = np.concatenate((i, ii))
            jj = Q[q] - 1
            j = np.concatenate((j, jj))
            W = np.abs(A[p[ii]] - A[p[jj]])  # Intensity derived weight
            s = np.concatenate((s, W))

            if iter_idx == 1:
                vertical_end = s.shape[0]  # Vertical edges length
            elif iter_idx == 5:
                diagonal_end = s.shape[0]  # Diagonal edges length

        # Normalize weights
        s = self.normalize(s)

        # Horizontal penalty
        s[:vertical_end] += gamma
        #s[vertical_end:diagonal_end] += gamma * np.sqrt(2) # --> In the paper it is sqrt(2) since the diagonal edges are longer yet does not exist in the original code

        # Normalize differences
        s = self.normalize(s)

        # Gaussian weighting function
        s = -(
            (np.exp(-beta * s, dtype="float64")) + 1.0e-6
        )  # --> This epsilon changes results drastically default: 1.e-6

        # Create Laplacian, diagonal missing
        L = csc_matrix((s, (i, j)))

        # Reset diagonal weights to zero for summing
        # up the weighted edge degree in the next step
        L.setdiag(0)

        # Weighted edge degree
        D = np.abs(L.sum(axis=0).A)[0]

        # Finalize Laplacian by completing the diagonal
        L.setdiag(D)

        return L
    
    def _solve_linear_system(self, D, rhs, tol=1.e-3, mode='bf'):

        if mode is not "octave":
            if mode is None:
                mode = 'cg_j'

            if mode == 'cg_mg' and not amg_loaded:
                print('Warning: "cg_mg" not available, it requires pyamg to be installed.')
                print('The "cg_j" mode will be used instead.')

                mode = 'cg_j'

            if mode == 'bf':
                X = spsolve(D, rhs.toarray()).T
            else:
                maxiter = None
                if mode == 'cg':
                    if UmfpackContext is None:
                        print('Warning: "cg" mode may be slow because UMFPACK is not available.')
                        print('Consider building Scipy with UMFPACK or use a ')
                        print('preconditioned version of CG ("cg_j" or "cg_mg" modes).')
                    M = None
                elif mode == 'cg_j':
                    M = sparse.diags(1.0 / D.diagonal())
                else:
                    lap_sparse = D.tocsr()
                    ml = ruge_stuben_solver(lap_sparse, coarse_solver='pinv')
                    M = ml.aspreconditioner(cycle='V')
                    maxiter = 30

                x, info = cg(D, rhs.toarray(), tol=tol, atol=0, M=M, maxiter=maxiter)
                
                if info > 0:
                    print("Warning: Conjugate gradient convergence to tolerance not achieved.")
                    print("Consider decreasing beta to improve system conditionning.")

                X = np.asarray(x)
        else:
            X = self.oc.mldivide(D, rhs)[:, 0]

        return X

    def confidence_estimation(self, A, seeds, labels, beta, gamma):
        """Compute confidence map

        Args:
            A (np.ndarray): Processed image.
            seeds (np.ndarray): Seeds for the random walks framework. These are indices of the source and sink nodes.
            labels (np.ndarray): Labels for the random walks framework. These represent the classes or groups of the seeds.
            beta: Random walks parameter that defines the sensitivity of the Gaussian weighting function.
            gamma: Horizontal penalty factor that adjusts the weight of horizontal edges in the Laplacian.

        Returns:
            map: Confidence map which shows the probability of each pixel belonging to the source or sink group.
        """

        # Index matrix with boundary padding
        G = np.arange(1, A.shape[0] * A.shape[1] + 1).reshape(A.shape[1], A.shape[0]).T
        pad = 1

        G = np.pad(G, (pad, pad), "constant", constant_values=(0, 0))
        B = np.pad(A, (pad, pad), "constant", constant_values=(0, 0))

        # Laplacian
        D = self.confidence_laplacian(G, B, beta, gamma)

        # Select marked columns from Laplacian to create L_M and B^T
        B = D[:, seeds]

        # Select marked nodes to create B^T
        N = np.sum(G > 0).item()
        i_U = np.setdiff1d(np.arange(N), seeds.astype(int))  # Index of unmarked nodes
        B = B[i_U, :]

        # Remove marked nodes from Laplacian by deleting rows and cols
        keep_indices = np.setdiff1d(np.arange(D.shape[0]), seeds)
        D = csc_matrix(D[keep_indices, :][:, keep_indices])

        # Define M matrix
        M = np.zeros((seeds.shape[0], 1), dtype="float64")
        M[:, 0] = labels == 1

        # Right-handside (-B^T*M)
        rhs = -B @ M  # type: ignore

        # Solve linear system
        x = self._solve_linear_system(D, rhs, tol=1.e-3)

        # Prepare output
        probabilities = np.zeros((N,), dtype="float64")
        # Probabilities for unmarked nodes
        probabilities[i_U] = x
        # Max probability for marked node
        probabilities[seeds[labels == 1].astype(int)] = 1.0

        # Final reshape with same size as input image (no padding)
        probabilities = probabilities.reshape((A.shape[1], A.shape[0])).T

        return probabilities

    def __call__(self, data: np.ndarray, downsample=None) -> np.ndarray:
        """Compute the confidence map

        Args:
            data (np.ndarray): RF ultrasound data (one scanline per column)

        Returns:
            map (np.ndarray): Confidence map
        """

        # Normalize data
        data = data.astype("float64")
        data = self.normalize(data)

        if self.mode == "RF":
            # MATLAB hilbert applies the Hilbert transform to columns
            data = np.abs(hilbert(data, axis=0)).astype("float64")  # type: ignore

        org_H, org_W = data.shape
        if downsample is not None:
            data = cv2.resize(data, (org_W // downsample, org_H // downsample), interpolation=cv2.INTER_CUBIC)

        seeds, labels = self.get_seed_and_labels(data, self.sink_mode, self.sink_mask)

        # Attenuation with Beer-Lambert
        W = self.attenuation_weighting(data, self.alpha)

        # Apply weighting directly to image
        # Same as applying it individually during the formation of the
        # Laplacian
        data = data * W

        # Find condidence values
        map_ = self.confidence_estimation(data, seeds, labels, self.beta, self.gamma)

        if downsample is not None:
            map_ = cv2.resize(map_, (org_W, org_H), interpolation=cv2.INTER_CUBIC)

        return map_

