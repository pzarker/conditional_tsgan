import numpy as np
import os
from typing import Sequence
from statsmodels.tsa.arima_process import arma_generate_sample
from torch.utils.data import Dataset


def get_data(data_type, data_options=None):
    """
    Helper/wrapper function to get the requested data.
    """
    labels = None
    pdf = None
    if data_type == 'load':
        data_dict = np.load(data_options).item()
        samples = data_dict['samples']
        pdf = data_dict['pdf']
        labels = data_dict['labels']
    elif data_type == 'sine':
        samples = sine_wave(**data_options)
    else:
        raise ValueError(data_type)
    print('Generated/loaded', len(samples), 'samples from data-type', data_type)
    return samples, pdf, labels


def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, **kwargs):
    ix = np.arange(seq_length) + 1
    samples = []
    for i in range(num_samples):
        signals = []
        for i in range(num_signals):
            f = np.random.uniform(low=freq_high, high=freq_low)     # frequency
            A = np.random.uniform(low=amplitude_high, high=amplitude_low)        # amplitude
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A*np.sin(2*np.pi*f*ix/float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_samples x seq_length x num_signals
    samples = np.array(samples)
    return samples


class ARMA(Dataset):

    def __init__(self, p: Sequence[float], q: Sequence[float], seed: int = None,
                 n_series: int = 200, n_points: int = 100, generate_new=False, save_csv=False):
        """
        Pytorch Dataset to sample a given ARMA process.

        y = ARMA(p,q)
        :param p: AR parameters
        :param q: MA parameters
        :param seed: random seed
        :param n_series: number of ARMA samples in your dataset
        :param datapoints: length of each sample
        """
        self.p = p
        self.q = q
        self.n_series = n_series
        self.n_points = n_points
        self.seed = seed
        if generate_new or (not os.path.isfile(self._get_filepath())):
            self.dataset = self._generate_ARMA()
            if save_csv:
                np.savetxt(self._get_filepath(), self.dataset, delimiter=',')
        else:
            self.dataset = np.loadtxt(open(self._get_filepath(), 'rb'), delimiter=',')

    def __len__(self):
        return self.n_series

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _get_filepath(self):
        filename = "".join((f"ar{len(self.p)}", "".join([f"_{format_num_fname(i)}" for i in self.p]),
                            f"_ma{len(self.q)}" + "".join([f"_{format_num_fname(i)}" for i in self.q]),
                            f'_N{self.n_series}_T{self.n_points}_seed{self.seed}.csv'))
        filepath = os.path.join(SIM_DIR, filename)
        return filepath


    def _generate_ARMA(self):

        if self.seed is not None:
            np.random.seed(self.seed)

        ar = np.r_[1, -np.array(self.p)]
        ma = np.r_[1, np.array(self.q)]
        burn = int(self.n_points / 10)

        dataset = [arma_generate_sample(ar=ar, ma=ma, nsample=self.n_points, burnin=burn) for _ in range(self.n_series)]
        return np.array(dataset)
