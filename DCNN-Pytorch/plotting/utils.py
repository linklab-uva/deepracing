import collections, collections.abc
import numpy as np
class PredictionResults(collections.abc.Mapping[str,np.ndarray]):
    def __init__(self, resultsdict : dict[str,np.ndarray], data_dir : str, modelname : str) -> None:
        self.resultsdict = resultsdict
        self.data_dir = data_dir
        self.modelname = modelname
    def __eq__(self, __other: 'PredictionResults') -> bool:
        return self.modelname == __other.modelname
    def __iter__(self):
        return iter(self.resultsdict)
    def __len__(self):
        return len(self.resultsdict)
    def __getitem__(self, key):
        return self.resultsdict[key]
    def __setitem__(self, key, value):
        self.resultsdict[key] = value
    def numsamples(self):
        return self.resultsdict["predictions"].shape[0]
    def keys(self):
        return self.resultsdict.keys()
    def subsample(self, idx : np.ndarray):
        resultsdict = {
            k : v[idx].copy() for (k,v) in self.resultsdict.items()
        }
        return PredictionResults(resultsdict, self.data_dir, self.modelname)
    def trim_percentiles(self, p0 : float = 25.0, pf : float = 75.0, whis : float = 1.5, metric : str = "ade"):
        err = self.resultsdict[metric]
        p0_value = np.percentile(err, p0)
        pf_value = np.percentile(err, pf)
        iqr = pf_value - p0_value
        maxval = p0_value + whis*iqr
        return err<=maxval

    @staticmethod
    def from_data_file(data_file : str, modelname : str, sort_idx : np.ndarray | None = None, allow_pickle=False) -> 'PredictionResults':
        data_dir = os.path.dirname(data_file)
        with open(data_file, "rb") as f:
            results_file = np.load(f, allow_pickle=allow_pickle)
            if sort_idx is None:
                results_dict = {k: v.copy() for (k,v) in results_file.items()}
            else:
                results_dict = {k: v[sort_idx].copy() for (k,v) in results_file.items()}
        return PredictionResults(results_dict, data_dir, modelname)
    def compute_fde(self):
        self.resultsdict["fde"] = np.linalg.norm(self.resultsdict["predictions"][:,-1,[0,1]] - self.resultsdict["ground_truth"][:,-1,[0,1]], ord=2.0, axis=1)