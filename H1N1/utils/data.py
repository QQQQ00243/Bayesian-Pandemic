import numpy as np
import pandas as pd
from scipy.stats import gamma

def compare_dates(date1, date2):
    '''
    Dtermine whether date1 is before date2, return True is date1 is before date2, False if not.
    date format: [year, month, day]
    '''
    for i1, i2 in zip(date1, date2):
        i = i1 - i2
        if i > 0:
            return False
        elif i < 0:
            return True
    return False


def parse_date_numeric(date: str):
    month, day, year = [int(i) for i in date.split(r"/")]
    return [year, month, day]


class H1N1:
    def __init__(
        self,
        shape=16,     # shape of the incubation period distribution (gamma)
        scale=0.125,  # scale of the incubation period distribution (gamma)
        tau=6,       # truncation bound of the incubation period distribution
        xlsx_file="./code/data/H1N1.xlsx" #xlsx file of the daily cases
    ):
        self.shape = shape
        self.scale = scale
        self.tau = tau
        self.xlsx_file = xlsx_file
        self.dates = np.array([])
        self.C = np.array([])
        self.I = np.array([])
        self.data = pd.DataFrame()
        self.data_seg = np.array([])

    def _parse_date(self, date):
        '''
        returns formatted date
        Example: date: 15th May 2009
                return: May 15
        '''
        day, month, year = date.split(" ")
        day = day[:-2]
        return f"{month} {day}"

    def _compute_serial_interval(self):
        si = [2.24, 2.85, 9.31, 49.83, 64.49, 49.07, 23.76, 7.07, 1.51]
        scale = (1.37 / 10.92 * 0.05 + 0.3) / 64.49
        self.beta = scale * np.array(si)
        # no return
        
    def _compute_time_delay(self):
        N = self.C.shape[0]
        H = np.zeros((N, N))
        psf = [gamma.pdf(i+1, a=self.shape, scale=self.scale) for i in range(self.tau)]
        psf = 1 / np.sum(psf) * np.array(psf)
        for i in range(N):
            if i+self.tau > N:
                temp = N
            else:
                temp = i+self.tau
            for j in range(i, temp):
                H[i, j] = psf[j-i]
        self.H = H
        # no return
    
    def load_data(self, n_padding=0):
        '''
        load data from outbreak
        n_padding: number of days of padding on the left with zero 
        '''
        data = pd.read_excel(self.xlsx_file)
        parsed_dates = [self._parse_date(date) for date in data["Date"]]
        data["Formatted date"] = parsed_dates

        day = int(parsed_dates[0].split(" ")[-1])
        if n_padding > day:
            raise ValueError(f"n_padding too large. Please keep the date in April by setting n_padding less than {day}")

        padding_dates = [f"April {day-i}" for i in range(n_padding, 0, -1)]

        self.C = np.concatenate([np.zeros((n_padding,)), data["Number of cases"]])
        self.dates = np.concatenate([padding_dates, parsed_dates])

        self._compute_serial_interval()
        self._compute_time_delay()

        # reconstruction of incidence sequence
        self.I = np.dot(self.H, self.C).astype(int)

        # prepare date segments for low-level model parameters estimation 
        n = len(self.beta)
        data_seg = np.empty((len(self.I)-n+1, n))
        for i in range(data_seg.shape[0]):
            data_seg[i, :] = self.I[i:i+n][::-1]
        self.dates = self.dates[n-1:]
        self.I = self.I[n-1:]
        self.C = self.C[n-1:]
        self.data_seg = data_seg


def main():
    pass


if __name__ == "__main__":
    main()

