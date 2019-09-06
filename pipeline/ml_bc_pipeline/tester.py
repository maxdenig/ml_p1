import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_bc_pipeline.data_loader import Dataset
from ml_bc_pipeline.data_preprocessing import Processor
from ml_bc_pipeline.feature_engineering import FeatureEngineer
from ml_bc_pipeline.model import grid_search_MLP, assess_generalization_auprc
from sklearn.datasets import make_classification, make_moons
from mlxtend.plotting import plot_decision_regions

def main():
    seed = 0
    np.random.seed(seed)
    X, y = make_classification(n_features=2, n_redundant=0, random_state=seed,
                               n_informative=2, n_clusters_per_class=1)
    X = X + np.random.uniform(-.5, .5, X.shape[0] * 2).reshape(X.shape)

    ds = pd.DataFrame(X, columns=["A", "B"])
    ds["Response"] = pd.Series(y)

    DF_train, DF_unseen = train_test_split(ds.copy(), test_size=0.2, stratify=ds["Response"],
                                           random_state=seed)

    #+++++++++++++++++ 5) modelling
    mlp_param_grid = {'mlpc__hidden_layer_sizes': [(3), (6), (3, 3), (5, 5)],
                      'mlpc__learning_rate_init': [0.001, 0.01]}

    mlp_gscv = grid_search_MLP(DF_train, mlp_param_grid, seed)
    print("Best parameter set: ", mlp_gscv.best_params_)
    # pd.DataFrame.from_dict(mlp_gscv.cv_results_).to_excel("D:\\PipeLines\\project_directory\\data\\mlp_gscv.xlsx")

    #+++++++++++++++++ 6) retraining & assessment of generalization ability
    auprc = assess_generalization_auprc(mlp_gscv.best_estimator_, DF_unseen)
    print("AUPRC: {:.2f}".format(auprc))

    plot_decision_regions(X=ds.iloc[:, :-1].values, y=ds.iloc[:, -1].values, clf=mlp_gscv.best_estimator_,
                          X_highlight=DF_unseen.iloc[:, :-1].values,
                          scatter_highlight_kwargs={'s': 120, 'label': 'Test data', 'alpha': 0.7})
    plt.show()

if __name__ == "__main__":
    main()