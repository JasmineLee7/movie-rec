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