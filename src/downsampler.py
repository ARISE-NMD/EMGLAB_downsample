import numpy as np
import pickle
import time
import gzip
import io
from pathlib import Path
from tsdownsample import MinMaxDownsampler, M4Downsampler, LTTBDownsampler, MinMaxLTTBDownsampler
from helpers import mark_done, mark_undone
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.stats import skew
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kurtosis as kurt
from scipy.stats import gaussian_kde
from scipy.signal import decimate

def f_interpolate(x, y):
    if len(x) == len(y):
        return x, y
    x_t = np.linspace(0, 2, len(x))
    y_t = np.linspace(0, 2, len(y))

    if len(x) > len(y):
        y = np.interp(x_t, y_t, y)
    else:
        x = np.interp(y_t, x_t, x)
    return x, y

def pearson(x, y):
    x, y = f_interpolate(x, y)
    corr, _ = pearsonr(x, y)
    return 1 - corr

def spearman(x, y):
    x, y = f_interpolate(x, y)
    corr, _ = spearmanr(x, y)
    return 1 - corr

def rmse(x, y):
    x, y = f_interpolate(x, y)
    return np.sqrt(mean_squared_error(x, y))

def nmse(x, y):
    x, y = f_interpolate(x, y)
    mse = mean_squared_error(x, y)
    return mse / np.var(x)

def envelope(signal):
    return np.abs(hilbert(signal))

def envelope_corr_pearsonr(x, y):
    x, y = f_interpolate(x, y)
    x_env = envelope(x)
    y_env = envelope(y)
    corr, _ = pearsonr(x_env, y_env)
    return 1 - corr

def zcr(x, y):
    x_zcr = np.sum(np.diff(np.sign(x)) != 0) / (len(x) - 1)
    y_zcr = np.sum(np.diff(np.sign(y)) != 0) / (len(y) - 1)
    return np.abs(x_zcr - y_zcr)

def skewness(x, y):
    x_skew = skew(x)
    y_skew = skew(y)
    return np.abs(x_skew - y_skew)

def kurtosis(x, y):
    x_kurt = kurt(x)
    y_kurt = kurt(y)
    return np.abs(x_kurt - y_kurt)

def peak_count(x, y):
    x_peaks, _ = find_peaks(x)
    y_peaks, _ = find_peaks(y)
    return np.abs(len(x_peaks) - len(y_peaks))

def compressed_size(data):
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as f:
        f.write(data)
    return len(out.getvalue())

def ncd(x, y):
    x_bytes = x.tobytes()
    y_bytes = y.tobytes()
    xy_bytes = x_bytes + y_bytes

    Cx = compressed_size(x_bytes)
    Cy = compressed_size(y_bytes)
    Cxy = compressed_size(xy_bytes)

    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)

def pdf_hist(x, y, bins=100):
    x = np.ravel(x)
    y = np.ravel(y)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    p_hist, bin_edges = np.histogram(x, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(y, bins=bins, range=(min_val, max_val), density=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, p_hist, q_hist

def pdf_kde(x, y, bandwidth=None, points=500):
    x = np.ravel(x)
    y = np.ravel(y)

    kde_x = gaussian_kde(x, bw_method=bandwidth)
    kde_y = gaussian_kde(y, bw_method=bandwidth)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    xs = np.linspace(min_val, max_val, points)

    return xs, kde_x(xs), kde_y(xs)

def jsd(x, y, method='kde', bandwidth=None, points=500, bins=100):
    if method not in ['kde', 'hist']:
        raise ValueError("Only available methods: 'kde' or 'hist' for pdf approximation")

    if method == 'kde':
        xs, p_pdf, q_pdf = pdf_kde(x, y, bandwidth=bandwidth, points=points)
    else:
        xs, p_pdf, q_pdf = pdf_hist(x, y, bins=bins)

    eps = 1e-12
    p_pdf = np.clip(p_pdf, eps, None)
    q_pdf = np.clip(q_pdf, eps, None)

    p_pdf /= np.sum(p_pdf)
    q_pdf /= np.sum(q_pdf)

    m_pdf = 0.5 * (p_pdf + q_pdf)
    dx = xs[1] - xs[0]

    kl_pm = np.sum(p_pdf * np.log(p_pdf / m_pdf)) * dx
    kl_qm = np.sum(q_pdf * np.log(q_pdf / m_pdf)) * dx

    return 0.5 * (kl_pm + kl_qm)

def euclidian_psd(x, y, fs=23437.5):
    f_x, Pxx_x = welch(x, fs=fs)
    f_y, Pxx_y = welch(y, fs=fs)

    # Interpolate Pxx_y onto f_x
    interp_y = interp1d(f_y, Pxx_y, kind="linear", fill_value="extrapolate")
    Pxx_y_resampled = interp_y(f_x)

    # Normalize
    Pxx_x /= np.sum(Pxx_x)
    Pxx_y_resampled /= np.sum(Pxx_y_resampled)

    return np.sqrt(np.sum((Pxx_x - Pxx_y_resampled) ** 2))

def measure_distance_metrics(X, Y, logger=None):
    if len(X) != len(Y):
        raise ValueError("X and Y must be the same size")

    timings = {}

    def timed_call(name, func, *args, **kwargs):
        start = time.perf_counter()
        val = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        timings.setdefault(name, []).append(elapsed)
        return val

    # Results
    pearson_dist, spearman_dist, rmse_dist, nmse_dist = [], [], [], []
    envelope_corr_pearsonr_dist, zcr_dist, skewness_dist, kurtosis_dist = [], [], [], []
    peak_count_dist, euclidian_psd_dist = [], []
    ncd_dist, jsd_hist_dist = [], []

    for x, y in zip(X, Y):
        pearson_dist.append(timed_call("pearson", pearson, x, y))
        spearman_dist.append(timed_call("spearman", spearman, x, y))
        rmse_dist.append(timed_call("rmse", rmse, x, y))
        nmse_dist.append(timed_call("nmse", nmse, x, y))
        envelope_corr_pearsonr_dist.append(timed_call("envelope_pearsonr_corr", envelope_corr_pearsonr, x, y))
        zcr_dist.append(timed_call("zcr", zcr, x, y))
        skewness_dist.append(timed_call("skewness", skewness, x, y))
        kurtosis_dist.append(timed_call("kurtosis", kurtosis, x, y))
        peak_count_dist.append(timed_call("peak_count", peak_count, x, y))
        euclidian_psd_dist.append(timed_call("euclidian_psd", euclidian_psd, x, y))
        ncd_dist.append(timed_call("ncd", ncd, x, y))
        jsd_hist_dist.append(timed_call("jsd_hist", jsd, x, y, method='hist'))

    res = {
        'pearson': np.array(pearson_dist),
        'spearman': np.array(spearman_dist),
        'rmse': np.array(rmse_dist),
        'nmse': np.array(nmse_dist),
        'envelope_pearsonr_corr': np.array(envelope_corr_pearsonr_dist),
        'zcr': np.array(zcr_dist),
        'skewness': np.array(skewness_dist),
        'kurtosis': np.array(kurtosis_dist),
        'peak_count_delta': np.array(peak_count_dist),
        'euclidian_psd': np.array(euclidian_psd_dist),
        'ncd': np.array(ncd_dist),
        'jsd': np.array(jsd_hist_dist),
    }

    return res

class Decimate:
    def downsample(self, x, n_out):
        q = len(x) // n_out
        if q <= 1:
            return x
        return decimate(x, q, ftype='fir', zero_phase=True)


class Downsampler:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.mapping = {
            "Original": None,
            "Decimate": Decimate,
            "LTTB": LTTBDownsampler,
            "MinMax": MinMaxDownsampler,
            "M4": M4Downsampler,
            "MinMaxLTTB": MinMaxLTTBDownsampler
        }

    def _ts_downsample(self, X, q, tsdownsampler):
            if q == 1 or tsdownsampler is None:
                return X
            n_out = len(X[0].squeeze()) // q
            if n_out % 2 != 0 and tsdownsampler is MinMaxDownsampler:
                n_out -= 1
            if n_out % 4 != 0 and tsdownsampler is M4Downsampler:
                n_out -= n_out % 4
                
            X_out = []
            
            for x in X.squeeze():
                s_ds = tsdownsampler().downsample(x, n_out=int(n_out))
                if tsdownsampler != Decimate:
                    X_out.append(x[s_ds])
                else:
                    X_out.append(s_ds) 
            return np.array(X_out)

    def run(self, input_path: Path, output_path: Path):
            mark_undone(output_path, "downsampling")
            self.logger.info("Running downsampling...")

            self.logger.info("Loading preprocessed data...")
            X = np.load(input_path / "X.npy")
            self.logger.info(f"Loaded preprocessed data X: {X.shape}")

            self.logger.info(f"Downsampling with {len(self.config.downsamplers) - 1} downsamplers and {len(self.config.downsampling_factors)} factors...")

            for downsampler in self.config.downsamplers:
                current_dir = output_path / downsampler
                current_dir.mkdir(parents=True, exist_ok=True)
                if downsampler == 'Original':
                    downsampling_factors = [1]
                else:
                    downsampling_factors = self.config.downsampling_factors

                for downsampling_factor in downsampling_factors:
                    X_ds = self._ts_downsample(X, downsampling_factor, self.mapping[downsampler])
                    self.logger.info(f"{downsampler}({downsampling_factor}), {X.shape} -> {X_ds.shape}")
                    
                    np.save(current_dir / f"X_{downsampling_factor}.npy", X_ds)

                    metrics = measure_distance_metrics(X, X_ds, self.logger)
                    with open(current_dir / f'measures_{downsampling_factor}.pkl', 'wb') as f:
                        pickle.dump(metrics, f)

            mark_done(output_path, "downsampling")
