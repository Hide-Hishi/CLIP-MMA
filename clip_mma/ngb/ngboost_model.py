import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import LogScore

class Regressor(NGBRegressor):
    """
    A specialized version of the NGBoost Regressor that includes functionality for stochastic sampling of data points,
    handling of categorical data via a lattice structure.

    Parameters
    ----------
    Dist : distribution from ngboost.distns (default Normal)
        The distribution of the response variable.
    Score : scoring rule from ngboost.scores (default LogScore)
        The scoring rule for the NGBoost algorithm.
    Base : base learner from ngboost.learners (default default_tree_learner)
        The base learner to use.
    natural_gradient : bool (default True)
        If True, use the natural gradient at each boosting step.
    n_estimators : int (default 500)
        The number of boosting stages that will be performed.
    learning_rate : float (default 0.01)
        The step size to use for each update of the model.
    minibatch_frac : float (default 1.0)
        The fraction of the training data used for each boosting stage.
    col_sample : float (default 1.0)
        The fraction of features to use per boosting stage.
    verbose : bool (default True)
        Whether to output progress messages to the console.
    verbose_eval : int (default 100)
        The interval in terms of boosting stages at which verbose output is provided.
    tol : float (default 1e-4)
        The tolerance for stopping.
    random_state : int or RandomState, optional
        The seed of the pseudo random number generator to use.
    validation_fraction : float (default 0.1)
        The proportion of the dataset to include in the validation split.
    early_stopping_rounds : int, optional
        The number of rounds without improvement after which training will be stopped.
    """

    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
    ):

        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
            validation_fraction,
            early_stopping_rounds,
        )

    def sample(self, X, Y, sample_weight, params, lattice_size):
        idxs = np.arange(len(Y))
        col_idx = np.arange(X.shape[1])
        #let lattice_size correspond to 2 sigma
        #sigma = lattice_size/2
        sigma = lattice_size

        if self.minibatch_frac != 1.0:
            sample_size = int(self.minibatch_frac * len(Y))
            idxs = self.random_state.choice(
                np.arange(len(Y)), sample_size, replace=False
            )

        if self.col_sample != 1.0:
            col_size = int(self.col_sample * X.shape[1])
            col_idx = self.random_state.choice(
                np.arange(X.shape[1]), col_size, replace=False
            )

        weight_batch = None if sample_weight is None else sample_weight[idxs]

        noise_arr = np.random.normal(0, sigma, Y[idxs].shape)
        #noise_arr = (np.random.rand(Y[idxs].shape[0],Y[idxs].shape[1])-0.5)*lattice_size
        Y_samples = Y[idxs] + noise_arr

        return (
            idxs,
            col_idx,
            X[idxs, :][:, col_idx],
            Y_samples,
            weight_batch,
            params[idxs, :],
        )

    def fit(
        self,
        X,
        Y,
        lattice_size,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):

        self.base_models = []
        self.scalings = []
        self.col_idxs = []

        return self.partial_fit(
            X,
            Y,
            lattice_size,
            X_val=X_val,
            Y_val=Y_val,
            sample_weight=sample_weight,
            val_sample_weight=val_sample_weight,
            train_loss_monitor=train_loss_monitor,
            val_loss_monitor=val_loss_monitor,
            early_stopping_rounds=early_stopping_rounds,
        )

    def partial_fit(
        self,
        X,
        Y,
        lattice_size,
        X_val=None,
        Y_val=None,
        sample_weight=None,
        val_sample_weight=None,
        train_loss_monitor=None,
        val_loss_monitor=None,
        early_stopping_rounds=None,
    ):

        if len(self.base_models) != len(self.scalings) or len(self.base_models) != len(
            self.col_idxs
        ):
            raise RuntimeError(
                "Base models, scalings, and col_idxs are not the same length"
            )

        # if early stopping is specified, split X,Y and sample weights (if given) into training and validation sets
        # This will overwrite any X_val and Y_val values passed by the user directly.
        if self.early_stopping_rounds is not None:

            early_stopping_rounds = self.early_stopping_rounds

            if sample_weight is None:
                X, X_val, Y, Y_val = train_test_split(
                    X,
                    Y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                )

            else:
                X, X_val, Y, Y_val, sample_weight, val_sample_weight = train_test_split(
                    X,
                    Y,
                    sample_weight,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                )

        if Y is None:
            raise ValueError("y cannot be None")

        X, Y = check_X_y(
            X, Y, accept_sparse=True, y_numeric=True, multi_output=self.multi_output
        )

        self.n_features = X.shape[1]
        loss_list = []
        self.fit_init_params_to_marginal(Y)

        params = self.pred_param(X)

        if X_val is not None and Y_val is not None:
            X_val, Y_val = check_X_y(
                X_val,
                Y_val,
                accept_sparse=True,
                y_numeric=True,
                multi_output=self.multi_output,
            )
            val_params = self.pred_param(X_val)
            val_loss_list = []
            best_val_loss = np.inf

        if not train_loss_monitor:
            train_loss_monitor = lambda D, Y, W: D.total_score(  # NOQA
                Y, sample_weight=W
            )

        if not val_loss_monitor:
            val_loss_monitor = lambda D, Y: D.total_score(  # NOQA
                Y, sample_weight=val_sample_weight
            )  # NOQA

        for itr in range(len(self.col_idxs), self.n_estimators + len(self.col_idxs)):
            _, col_idx, X_batch, Y_batch, weight_batch, P_batch = self.sample(
                X, Y, sample_weight, params, lattice_size
            )
            self.col_idxs.append(col_idx)

            D = self.Manifold(P_batch.T)

            loss_list += [train_loss_monitor(D, Y_batch, weight_batch)]
            loss = loss_list[-1]
            grads = D.grad(Y_batch, natural=self.natural_gradient)

            proj_grad = self.fit_base(X_batch, grads, weight_batch)
            scale = self.line_search(proj_grad, P_batch, Y_batch, weight_batch)

            # pdb.set_trace()
            params -= (
                self.learning_rate
                * scale
                * np.array([m.predict(X[:, col_idx]) for m in self.base_models[-1]]).T
            )

            val_loss = 0
            if X_val is not None and Y_val is not None:
                val_params -= (
                    self.learning_rate
                    * scale
                    * np.array(
                        [m.predict(X_val[:, col_idx]) for m in self.base_models[-1]]
                    ).T
                )
                val_loss = val_loss_monitor(self.Manifold(val_params.T), Y_val)
                val_loss_list += [val_loss]
                if val_loss < best_val_loss:
                    best_val_loss, self.best_val_loss_itr = val_loss, itr
                if (
                    early_stopping_rounds is not None
                    and len(val_loss_list) > early_stopping_rounds
                    and best_val_loss
                    < np.min(np.array(val_loss_list[-early_stopping_rounds:]))
                ):
                    if self.verbose:
                        print("== Early stopping achieved.")
                        print(
                            f"== Best iteration / VAL{self.best_val_loss_itr} (val_loss={best_val_loss:.4f})"
                        )
                    break

            if (
                self.verbose
                and int(self.verbose_eval) > 0
                and itr % int(self.verbose_eval) == 0
            ):
                grad_norm = np.linalg.norm(grads, axis=1).mean() * scale
                print(
                    f"[iter {itr}] loss={loss:.4f} val_loss={val_loss:.4f} scale={scale:.4f} "
                    f"norm={grad_norm:.4f}"
                )

            if np.linalg.norm(proj_grad, axis=1).mean() < self.tol:
                if self.verbose:
                    print(f"== Quitting at iteration / GRAD {itr}")
                break

        self.evals_result = {}
        metric = self.Score.__name__.upper()
        self.evals_result["train"] = {metric: loss_list}
        if X_val is not None and Y_val is not None:
            self.evals_result["val"] = {metric: val_loss_list}

        return self