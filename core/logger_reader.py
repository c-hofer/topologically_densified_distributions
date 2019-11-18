import json
import pickle
import glob
import torch
import datetime
from pathlib import Path
from collections import Counter, defaultdict


class LoggerReader:
    _args_file_name = 'args.json'
    _module_ext = '.pth'
    _pickle_ext = '.pkl'

    def __init__(self, folder_path: Path):
        self.path = Path(folder_path)

        self._values_by_run = None

        with open(self.path / self._args_file_name, 'r') as fid:
            self._experiment_args = json.load(fid)

    def load_model(self, run, key):
        p = self.path / (str(run) + '/' + key + self._module_ext)
        return torch.load(p)

    def _get_run_folder_paths(self):
        folders = [x for x in self.path.iterdir() if x.is_dir()]

        if len(folders) > 0:

            # check for validity...
            folders_tmp = [int(x.name) for x in folders]
            Counter(folders_tmp) == Counter(range(1, max(folders_tmp)+1))

        return folders

    @property
    def experiment_args(self):
        return dict(self._experiment_args)

    @property
    def values_by_run(self):
        if self._values_by_run is None:
            tmp = defaultdict(dict)

            folders = self._get_run_folder_paths()

            for fd in folders:
                files = glob.glob(str(fd) + '/*' + self._pickle_ext)
                files = [Path(x) for x in files]

                for fl in files:
                    with open(fl, 'br') as fid:
                        v = pickle.load(fid)

                    tmp[int(fd.name)][fl.name.split('.')[0]] = v

            self._values_by_run = dict(tmp)

        return self._values_by_run

    def get_value(self, run, key):
        pth = self.path / (str(run) + '/' + key + self._pickle_ext)
        with open(pth, 'br') as fid:
            return pickle.load(fid)

    @property
    def progress(self):
        folders = self._get_run_folder_paths()
        i_last_run = len(folders)

        if i_last_run == 0:
            return None

        else:
            epoch_i = self.get_value(i_last_run, 'epoch_i')[-1]
            return (i_last_run, epoch_i)

    @property
    def date(self):
        time_sig = str(self.path.name).split('__')[0]
        time_sig = time_sig.split('-')
        time_sig = [int(x) for x in time_sig]

        return datetime.datetime(
            year=time_sig[2],
            month=time_sig[0],
            day=time_sig[1],
            hour=time_sig[3],
            minute=time_sig[4],
            second=time_sig[5]
        )
