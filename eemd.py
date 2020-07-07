from emd import EMD
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EEMD():
    def __init__(self, criterion="num_sifts", num_sifts=100, sd=0.2, ensemble_size=500, snr=20):
        self.criterion = criterion
        self.num_sifts = num_sifts
        self.sd = sd
        self.ensemble_size = ensemble_size
        self.snr = snr

    def __call__(self, s, modal=True, concurrent=True):
        if concurrent:
            eemd_set = []
            with ProcessPoolExecutor() as executor:
                eemd_futures = [executor.submit(ensemble_imf, s, self.criterion, self.num_sifts, self.sd, self.snr)
                                for _ in range(self.ensemble_size)]
                for future in as_completed(eemd_futures):
                    eemd_set.append(future.result())

        if not concurrent:
            eemd_set = [ensemble_imf(s, self.criterion, self.num_sifts, self.sd, self.snr)
                        for _ in range(self.ensemble_size)]

        if modal:
            num_imf = {}
            for imf_set in eemd_set:
                n_imf = len(imf_set)
                if n_imf in num_imf:
                    num_imf[n_imf] += 1
                else:
                    num_imf[n_imf] = 1
            modal_imf = max(num_imf, key=num_imf.get)
            eemd_set = np.asarray([imf_set for imf_set in eemd_set if len(imf_set) == modal_imf])
            eemd_set = np.mean(eemd_set, axis=0)
        return eemd_set


def ensemble_imf(s, criterion, num_sifts, sd, snr):
    noise = gen_noise(s, snr)
    emd = EMD(criterion=criterion, num_sifts=num_sifts, sd=sd)
    imf_set = emd(s + noise)
    return imf_set


def signal_power(s):
    return np.mean(np.square(s))


def gen_noise(s, target_snr):
    noise = np.random.normal(0, 1, s.shape)
    s_power = signal_power(s)
    n_power = signal_power(noise)
    k = (s_power / n_power) * 10 ** (-target_snr / 10)
    scale_noise = np.sqrt(k) * noise
    return scale_noise


if __name__ == "__main__":
    df = pd.read_csv('daily-min-temperatures.csv')
    s = df.values[:, 1].astype('float')
    eemd = EEMD(snr=10)
    imf_set = eemd(s)

    fig, axs = plt.subplots(len(imf_set) + 1, 1)
    axs[0].plot(s)
    for i in range(len(imf_set)):
        axs[i + 1].plot(imf_set[i])
    plt.show()

    plt.plot(np.sum(imf_set, axis=0)[:250])
    plt.plot(s[:250])
    plt.show()
