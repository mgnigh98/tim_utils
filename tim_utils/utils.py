import sklearn.mixture

from . import *
from copy import deepcopy

from .kde import KernelDensity
from .numpy import array

import traceback

bool = [False, True]

def clip(clip_text):
    subprocess.run("clip", input=clip_text, text=True)

def set_plt():
    font = {'family' : 'Tahoma'}
    plt.rc('font', **font)
    np.set_printoptions(suppress=True)

def torch_device():
    return (
    "cuda"
    if T.cuda.is_available()
    else "mps"
    if T.backends.mps.is_available()
    else "cpu"
)

def torch_parameter_count(net, only_trainable=True):
    if only_trainable:
        params = sum(p.numel() for p in net.parameters() if p.requires_grad )
    else:
        params = sum(p.numel() for p in net.parameters() )
    print(f"The network has {params} trainable parameters")


def find_nClusters(data, max_clusters=20, cluster_method="kmeans", spectral_affinity="rbf", cov_type="diag",
                   min_points_per_cluster=3):
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

    cluster = SimpleNamespace(**{})

    cluster.score = -1
    cluster.labels = np.repeat(1, data.shape[0])
    cluster.n = 1
    cluster.model = None
    for n_clusters in range(2, max_clusters+1):
        if cluster_method == "kmeans":
            clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        elif cluster_method == "spectral":
            clusterer = sklearn.cluster.SpectralClustering(n_clusters=n_clusters, n_init=10, random_state=0,
                                                       affinity=spectral_affinity)
        elif cluster_method == "bgmm":
            clusterer = sklearn.mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type=cov_type,
                                                            random_state=0)

        else:
            raise Exception("Cluster method not in list of defined methods ['kmeans', 'spectral', 'bgmm']")
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = sklearn.metrics.silhouette_score(data, cluster_labels)
        if (silhouette_avg > cluster.score) and (np.unique(cluster_labels,
                                                           return_counts=True)[1]>=min_points_per_cluster).all():
            cluster.score = silhouette_avg
            cluster.n = n_clusters
            cluster.labels = cluster_labels
            cluster.model = clusterer
    return cluster


class plotlyTrace:
    def __init__(self, dims=3, graph_type ="scatter"):
        self.dims = dims
        self.graph_type = graph_type
        self.traces = []

    def add_trace(self,x, color=None, name=''):
        autocolorscale = True if color else False
        if self.dims == 3:
            component1, component2, component3 = x[:, 0], x[:, 1], x[:, 2]
            if self.graph_type == "scatter":
                trace = go.Scatter3d(x=component1,
                                     y=component2,
                                     z=component3,
                                     mode='markers',
                                     marker=dict(autocolorscale=autocolorscale, color=color),
                                     name=name
                                     )
            elif self.graph_type == "line":
                trace = go.Line(x=component1,
                                y=component2,
                                z=component3,
                                mode='markers',
                                marker=dict(autocolorscale=autocolorscale, color=color),
                                name=name
                                )
        elif self.dims == 2:
            component1, component2 = x[:, 0], x[:, 1]
            if self.graph_type == "scatter":
                trace = go.Scatter(x=component1, y=component2, mode='markers',
                                   marker=dict(autocolorscale=autocolorscale, color=color),
                                   name=name)
        self.traces.append(trace)

    def plot(self):
        return go.Figure(data=self.traces)


def act_func(activation):
    return T.nn.ModuleDict([
        ['relu', T.nn.ReLU(inplace=True)],
        ['leaky_relu', T.nn.LeakyReLU(negative_slope=0.2, inplace=True)],
        ['selu', T.nn.SELU(inplace=True)],
        ['none', T.nn.Identity()]
    ])[activation]


class oldSpecScaler:
    def __init__(self, specs, tighten_specs=False, tighten_method='absolute'):
        self.specs = specs
        self.tighten_specs = tighten_specs
        if tighten_method not in ['absolute', 'relative']:
            raise NotImplementedError("tighten_method must be one of 'absolute', 'relative'")
        else:
            self.tighten_method = tighten_method

    def _tighten_specs(self, lower, upper, col):
        if self.tighten_method == 'absolute':
            mean = np.mean([lower, upper])
            range = upper-lower

            lower = mean - self.tighten_specs*(range/2)
            upper = mean + self.tighten_specs*(range/2)
        elif self.tighten_method == 'relative':
            print(type(self.data))
            print(col)
            data = self.data[col]
            raise Exception
            lower = np.quantile(data[(data>lower) & (data<upper)], (1-self.tighten_specs)/2)
            upper = np.quantile(data[(data>lower) & (data<upper)], 1-((1-self.tighten_specs)/2))
        return lower, upper
        
        

    def _upper_lower(self, column):
        mask_ = self.specs[self.specs.iloc[:, 0].apply(lambda x: x in column)]
        try:
            mask = mask_.iloc[[mask_.iloc[:, 0].str.len().argmax()]]
            lower = mask[[col for col in mask.columns if "low" in col.lower()][0]].values.item()
            upper = mask[
                [col for col in mask.columns if ("high" in col.lower()) | ("up" in col.lower())][0]].values.item()
            if self.tighten_specs:
                lower, upper  = self._tighten_specs(lower, upper, column)
        except Exception as e:
            if str(e) == 'attempt to get argmax of an empty sequence':
                print(f"Warning: column '{column}' not found in specification file, ignore if not expected to scale column")
            else:
                print("EXCEPTION")
                print(e)
                traceback.print_exc()
            upper, lower = np.NaN, np.NaN
        return upper, lower

    def upper_lower_limits(self, columns):
        columns = [columns] if isinstance(columns, str) else columns
        lowers, uppers = np.array([]), np.array([])
        for col in columns:
            upper, lower = self._upper_lower(col)
            lowers = np.append(lowers, lower)
            uppers = np.append(uppers, upper)
        return uppers, lowers

    def transform(self, data):
        self.data = data
        scaled = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            data_col = self.data[col]
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                scaled[col] = self.data[col]
            else:
                scaled[col] = (data_col - lower) / (upper-lower)
        return scaled
    fit_transform = transform

    def inverse_transform(self, scaled):
        try:
            if "pandas" in str(type(scaled)):
                in_scaled = pd.DataFrame(columns=scaled.columns)
            else:
                in_scaled = pd.DataFrame(columns=self.data.columns)
        except:
            raise Exception("'transform' was not called previously and input to 'inverse_transform' did not contain column labels")

        for i,col in enumerate(self.data.columns):
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                in_scaled[col] = scaled[col]
            elif "numpy" in str(type(scaled)):
                in_scaled[col] = scaled[:,i]*(upper-lower)+lower
            elif "pandas" in str(type(scaled)):
                in_scaled[col] = scaled[col]*(upper-lower)+lower
            else:
                raise Exception("Ensure to use Pandas DataFrame or Numpy Array as input")
        return in_scaled

    def return_stats(self, data):
        stats = pd.DataFrame(columns=["Parameter", "Above", "Below", "Within", "UpperLimit", "LowerLimit", "Max", "Min"])
        stats["Parameter"] = data.columns
        total_count = data.shape[0]
        for i,col in enumerate(data.columns):
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                stats.loc[stats.index[i],1:-2] = np.NaN
            else:
                stats.loc[stats.index[i], "UpperLimit":"Min"] = upper, lower, data[col].max(), data[col].min()
                above, below = np.sum(data[col]>upper), np.sum(data[col]<lower)
                stats.loc[stats.index[i], "Above":"Within"] = above ,below ,total_count-(above+below)
        return stats

class SpecScaler:
    def __init__(self, specs, tighten_specs=False, tighten_method='absolute'):
        self.specs = specs
        self.tighten_specs = tighten_specs
        if tighten_method not in ['absolute', 'relative']:
            raise NotImplementedError("tighten_method must be one of 'absolute', 'relative'")
        else:
            self.tighten_method = tighten_method

    def _tighten_specs(self, lower, upper, col):
        if self.tighten_method == 'absolute':
            mean = np.mean([lower, upper])
            range = upper-lower

            lower = mean - self.tighten_specs*(range/2)
            upper = mean + self.tighten_specs*(range/2)
        elif self.tighten_method == 'relative':
            data = self.data[col]
            lower = np.quantile(data[(data>lower) & (data<upper)], (1-self.tighten_specs)/2)
            upper = np.quantile(data[(data>lower) & (data<upper)], 1-((1-self.tighten_specs)/2))
        return lower, upper
        

    def _upper_lower(self, column):
        mask_ = self.specs[self.specs.iloc[:, 0].apply(lambda x: x in column)]
        try:
            mask = mask_.iloc[[mask_.iloc[:, 0].str.len().argmax()]]
            lower = mask[[col for col in mask.columns if "low" in col.lower()][0]].values.item()
            upper = mask[
                [col for col in mask.columns if ("high" in col.lower()) | ("up" in col.lower())][0]].values.item()
            if self.tighten_specs:
                lower_, upper_  = self._tighten_specs(lower, upper, column)
            else:
                lower_, upper_ = lower, upper
        except Exception as e:
            if str(e) == 'attempt to get argmax of an empty sequence':
                print(f"Warning: column '{column}' not found in specification file, ignore if not expected to scale column")
            else:
                print("EXCEPTION")
                print(e)
            upper_, lower_ = np.nan, np.nan
        return upper_, lower_

    def upper_lower_limits(self, columns):
        columns = [columns] if isinstance(columns, str) else columns
        lowers, uppers = np.array([]), np.array([])
        for col in columns:
            upper, lower = self._upper_lower(col)
            lowers = np.append(lowers, lower)
            uppers = np.append(uppers, upper)
        return uppers, lowers

    def transform(self, data):
        self.data = data
        scaled = pd.DataFrame(columns=self.data.columns)
        for col in self.data.columns:
            data_col = self.data[col]
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                scaled[col] = self.data[col]
            else:
                scaled[col] = (data_col - lower) / (upper-lower)
        return scaled
    fit_transform = transform

    def inverse_transform(self, scaled):
        try:
            if "pandas" in str(type(scaled)):
                in_scaled = pd.DataFrame(columns=scaled.columns)
            else:
                in_scaled = pd.DataFrame(columns=self.data.columns)
        except:
            raise Exception("'transform' was not called previously and input to 'inverse_transform' did not contain column labels")

        for i,col in enumerate(self.data.columns):
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                in_scaled[col] = scaled[col]
            elif "numpy" in str(type(scaled)):
                in_scaled[col] = scaled[:,i]*(upper-lower)+lower
            elif "pandas" in str(type(scaled)):
                in_scaled[col] = scaled[col]*(upper-lower)+lower
            else:
                raise Exception("Ensure to use Pandas DataFrame or Numpy Array as input")
        return in_scaled

    def return_stats(self, data):
        stats = pd.DataFrame(columns=["Parameter", "Above", "Below", "Within", "UpperLimit", "LowerLimit", "Max", "Min"])
        stats["Parameter"] = data.columns
        total_count = data.shape[0]
        for i,col in enumerate(data.columns[data.columns.isin(self.data.columns)]):
            upper, lower = self._upper_lower(col)
            if np.isnan(upper):
                stats.loc[stats["Parameter"]==col, 1:-2] = np.NaN
            else:
                stats.loc[stats["Parameter"]==col, "UpperLimit":"Min"] = upper, lower, data[col].max(), data[col].min()
                above, below = np.sum(data[col]>upper), np.sum(data[col]<lower)
                stats.loc[stats["Parameter"]==col, "Above":"Within"] = above ,below ,total_count-(above+below)
        return stats


def dir_files_list(directory):
    return [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

def gen_(n, template=None):
    if template is None:
        template = []
    for _ in range(n):
        yield deepcopy(template)

class log_print:
    def __init__(self, log_file, sys_stdout):
        self.log = log_file
        self.sys = sys_stdout
    def write(self, *args, **kwargs):
        self.log.write(*args, **kwargs)
        self.sys.write(*args, **kwargs)



