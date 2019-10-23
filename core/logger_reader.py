import json
import pickle
import glob
import dateutil
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

    @property
    def experiment_args(self):
        return dict(self._experiment_args)

    @property
    def values_by_run(self):
        if self._values_by_run is None:
            tmp = defaultdict(dict)

            folders = [x for x in self.path.iterdir() if x.is_dir()]

            # check for validity...
            folders_tmp = [int(x.name) for x in folders]
            Counter(folders_tmp) == Counter(range(1, max(folders_tmp)+1))

            for fd in folders:
                files = glob.glob(str(fd) + '/*' + self._pickle_ext)
                files = [Path(x) for x in files]

                for fl in files:
                    with open(fl, 'br') as fid:
                        v = pickle.load(fid)

                    tmp[int(fd.name)][fl.name.split('.')[0]] = v

            self._values_by_run = dict(tmp)

        return self._values_by_run

    @property
    def date(self):
        return dateutil.parser.parse(str(self.path.name).split('__')[0], ignoretz=True)
