import os
import json
import optuna
import bayesloop as bl

from datetime import datetime
from optuna.samplers import TPESampler
import sys
sys.path.append(r"c:\\Users\\qiyah\\Desktop\\Cam\\Pandemic\\code\\H1N1")
from utils.data import H1N1
from utils.model import likelihood, customOM
from bayesloop.transitionModels import SerialTransitionModel, GaussianRandomWalk, CombinedTransitionModel, BreakPoint


class Objective:
    def __init__(self, data_seg):
        self.data_seg = data_seg

    def __call__(self, trial):
        S = bl.Study(silent=True)
        S.loadData(self.data_seg, silent=True)
        L = customOM(likelihood, "R", bl.oint(0, 10, 100), "D", bl.cint(0, 8, 9))
        s1R = trial.suggest_float('s1R', low=0.1, high=3.0, step=0.1)
        s1D = trial.suggest_float('s1D', low=0.1, high=3.0, step=0.1)
        s2R = trial.suggest_float('s2R', low=0.1, high=3.0, step=0.1)
        s2D = trial.suggest_float('s2D', low=0.1, high=3.0, step=0.1)
        s3R = trial.suggest_float('s3R', low=0.1, high=3.0, step=0.1)
        s3D = trial.suggest_float('s3D', low=0.1, high=3.0, step=0.1)

        t1 = trial.suggest_int('t1', low=0, high=30, step=1)
        t2 = trial.suggest_int('t2', low=t1+1, high=31, step=1)

        T = SerialTransitionModel(
            CombinedTransitionModel(
                GaussianRandomWalk('s1R', s1R, target='R'),
                GaussianRandomWalk('s1D', s1D, target='D'),
            ),
            BreakPoint("t1", t1),
            CombinedTransitionModel(
                GaussianRandomWalk('s2R', s2R, target='R'),
                GaussianRandomWalk('s2D', s2D, target='D'),
            ),
            BreakPoint("t2", t2),
            CombinedTransitionModel(
                GaussianRandomWalk('s3R', s3R, target='R'),
                GaussianRandomWalk('s3D', s3D, target='D'),
            ),
        )
        S.set(L, T, silent=True)
        S.fit(evidenceOnly=True, silent=True)
        return S.log10Evidence


def main():
    # prepare log
    logs_dir = "./code/H1N1/logs/"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # load data
    h1n1 = H1N1()
    h1n1.load_data(n_padding=5)

    # create study
    study = optuna.create_study(sampler=TPESampler(), direction='maximize')
    obj = Objective(data_seg=h1n1.data_seg)
    study.optimize(obj, n_trials=100, n_jobs=5)

    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S").replace(":", "-")
    file = os.path.join(logs_dir, f"BestHyperParams-{now}.json")
    print(f"Saving best parameters to {file}")
    with open(file, "w") as f:
        json.dump(study.best_params, f, indent=4)


if __name__ == "__main__":
    main()
    
    