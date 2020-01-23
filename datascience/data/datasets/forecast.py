from torch.utils.data import Dataset


class ForecastDataset(Dataset):

    def __init__(self, dates_ref, runs_ref, time_steps_range):
        super(Dataset, self).__init__()

        self.data = []

        for date_ref in dates_ref:
            for run_ref in runs_ref:

                numpy_run_gfs = build_run(date_ref=date_ref, run_ref=run_ref, time_steps_range=time_steps_range)
                self.data.append(numpy_run_gfs)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.__sizeof__()


if __name__ == "__main__":

    dates_ref = ["20181216"]
    #runs_ref = [0, 6, 12, 18]
    runs_ref = [0, 6]
    time_steps_range = range(0, 13, 3)

    dataset = ForecastDataset(dates_ref=dates_ref, runs_ref=runs_ref, time_steps_range=time_steps_range)

    numpy_grib_run = dataset.__getitem__(1)
    numpy_grib_run.print_wind()
    numpy_grib_run.print_mask()

    print(numpy_grib_run.forecast.shape)
    print(numpy_grib_run.reftime.shape)