import numpy as np

def bias_baseline(R):
  # Overall mean plus one user effect and one movie effect.
  known = ~np.isnan(R)
  mu = np.nanmean(R)

  b_user = np.nanmean(R - mu, axis=1)
  residual = R - mu - b_user[:, None]

  n_item = known.sum(axis=0)
  b_item = np.zeros(R.shape[1])
  has_rating = n_item > 0
  b_item[has_rating] = (
      np.nansum(residual, axis=0) / n_item[has_rating])

  return mu, b_user, b_item



def baseline_matrix(R):
    """Full matrix: overall mean plus user and movie effects."""
    mu, b_user, b_item = bias_baseline(R)
    return mu + b_user[:, None] + b_item[None, :]


def rank_k(U, s, Vt, k):
    """Build a rank-k approximation from an SVD."""
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]


def iterative_svd_predict(R, k, n_iter=10):
    """Predict ratings by repeatedly estimating missing deviations."""
    known = ~np.isnan(R)
    B = baseline_matrix(R)
    X = np.where(known, R - B, 0.0)

    for _ in range(n_iter):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = rank_k(U, s, Vt, k)
        X = np.where(known, R - B, L)

    return L + B
