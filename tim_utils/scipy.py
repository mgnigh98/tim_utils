def patch_beta_fitstart():
    from scipy.stats._continuous_distns import beta_gen
    from scipy import optimize
    import numpy as np

    try:
        from scipy.stats._distn_infrastructure import CensoredData
    except ImportError:
        CensoredData = None

    def _skew(x):
        return ((x - x.mean()) ** 3).mean() / (x.std() ** 3)

    def _kurtosis(x):
        return ((x - x.mean()) ** 4).mean() / (x.std() ** 4) - 3

    def patched_fitstart(self, data):
        if CensoredData and isinstance(data, CensoredData):
            data = data._uncensor()

        g1 = _skew(data)
        g2 = _kurtosis(data)

        def func(x):
            a, b = x
            if a <= 0 or b <= 0:
                return [np.inf, np.inf]
            try:
                sk = 2 * (b - a) * np.sqrt(a + b + 1) / ((a + b + 2) * np.sqrt(a * b))
                ku = (
                    a ** 3
                    - a ** 2 * (2 * b - 1)
                    + b ** 2 * (b + 1)
                    - 2 * a * b * (b + 2)
                )
                ku /= a * b * (a + b + 2) * (a + b + 3)
                ku *= 6
                return [sk - g1, ku - g2]
            except Exception:
                return [np.inf, np.inf]

        result = optimize.least_squares(
            func,
            x0=(1.0, 1.0),
            bounds=([1e-3, 1e-3], [10, 10])
        )
        a, b = result.x
        return super(beta_gen, self)._fitstart(data, args=(a, b))

    # Apply the patch
    beta_gen._fitstart = patched_fitstart
