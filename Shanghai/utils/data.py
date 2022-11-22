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


def parse_date(date: str):
    # example date: 08/17/22 or 08/17/2022
    month, day, year = date.split("/")
    monthDict = {"1": "Jan.", "2": "Feb.", "3": "March", "4": "April", "5": "May", "6": "June", "7": "July", "8": "Aug.", "9": "Sept.", "10": "Oct.", "11": "Nov.", "12": "Dec."}
    return f"{monthDict[month]} {day}"


class Shanghai:
    def __init__(
        self,
        mean=2.72,  # mean of serial interval distribution (gamma)
        shape=3.25, # shape of serial interval distribution (gamma)
        scale=0.84, # scale of serial interval distribution (gamma)
        n_beta=12,  # number of discretized bins of serial interval distribution
        URL="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",    
    ):
        self.URL = URL
        self.mean = mean
        self.shape = shape
        self.scale = scale
        self.beta = np.empty((n_beta, ))
        self.dates = np.array([])
        self.C = np.array([])
        self.I = np.array([])
        self.data = pd.DataFrame()
        self.data_seg = np.array([])

    def _compute_serial_interval(self):
        for i in range(self.beta.shape[0]):
            self.beta[i] = gamma.pdf(i+1, a=self.shape, scale=self.scale) #, loc=self.shape, scale=self.scale)
        # no return
        
    def _compute_time_delay(self):
        N = self.C.shape[0]
        H = np.zeros((N, N))
        timepoint = [22, 3, 28]
        mean1 = 4
        mean2 = 2 
        for i in range(N):
            date = parse_date_numeric(self.data.index[i])
            if compare_dates(date, timepoint):
                mean = mean1
            else:
                mean = mean2
            for j in range(i, N):
                H[i, j] = gamma.pdf(j-i+1, mean)
        self.H = H
        # no retuen 
    
    def load_data(self, start=755, end=869):
        if len(self.C) != 0:
            print("[Warning] Data already loaded!")
        else:
            df = pd.read_csv(self.URL, on_bad_lines="skip")
            df_sh = df[df["Province/State"]=="Shanghai"]
            cases_sh = df_sh.iloc[:, 4:]
            data = pd.DataFrame(
                data=cases_sh.values.reshape((-1, 1)),
                index=cases_sh.columns,
                columns=["Total Confirmed Cases"]
            )
            data.index.name = "Date"
            data["Daily Confirmed Cases"] = data["Total Confirmed Cases"].diff(1)
    
            # clean negative data
            T = data["Total Confirmed Cases"].to_list()
            for i in [np.argwhere((data["Daily Confirmed Cases"] < 0).to_list()).item()]:
                mean = (T[i-1] + T[i+1]) // 2
                data.at[data.index[i], "Total Confirmed Cases"] = mean
            data["Daily Confirmed Cases"] = data["Total Confirmed Cases"].diff(1)

            # use data from Feb. 15, 2022 to June 8, 2022
            self.data = data.iloc[start:end, :]
            self.dates = np.array([parse_date(date) for date in self.data.index])
            self.C = self.data["Daily Confirmed Cases"].to_numpy()

            # replace outlier over 5000 with the mean of adjacent values
            idx = np.argwhere(self.C > 5000)
            self.C[idx] = (self.C[idx-1] + self.C[idx+1]) // 2

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


from scipy.stats import gamma


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
    
    def load_data(self, n_padding=10):
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
    sh = Shanghai()
    sh.load_data()

    print(len(sh.dates))


if __name__ == "__main__":
    main()

