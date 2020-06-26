from emd import EMD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

    
class EEMD():
    def __init__(self, criterion="num_sifts", num_sifts=100, sd=0.2, ensemble_size=100, snr=34):
        self.criterion = criterion
        self.num_sifts = num_sifts
        self.sd = sd
        self.ensemble_size = ensemble_size
        self.snr = snr

    def __call__(self, s, concurrent=True):
        with ProcessPoolExecutor() as executor:
           eemd_set = [executor.submit(ensemble_imf, s, self.criterion, self.num_sifts, self.sd, self.snr)
                   for _ in range(self.ensemble_size)]
        if not concurrent:
            eemd_set = [ensemble_imf(s, self.criterion, self.num_sifts, self.sd, self.snr)
                    for _ in range(self.ensemble_size)]
        return eemd_set

def ensemble_imf(s, criterion, num_sifts, sd, snr):
    noise = gen_noise(s, snr)
    emd = EMD(criterion=criterion, num_sifts=num_sifts, sd=sd)
    imf_set = emd(s + noise)
    return imf_set

def signal_power(s):
    return np.mean(np.square(s))

def gen_noise(s, target_snr=34):
    noise = np.random.normal(0, 1, s.shape)
    s_power = signal_power(s)
    n_power = signal_power(noise)
    k = (s_power / n_power) * 10 ** (-target_snr / 10)
    scale_noise = np.sqrt(k) * noise
    return scale_noise

if __name__ == "__main__":
    df = pd.read_csv('daily-min-temperatures.csv')
    s = df.values[:, 1].astype('float')
    eemd = EEMD(ensemble_size=50)
    eemd_set = eemd(s, False)
