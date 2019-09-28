"""Object representation of a fit PsyTrack model."""
from copy import deepcopy
import numpy as np
import os.path as opath
import yaml

from flow import config, paths
from flow.misc import loadmat, matlabifypars, mkdir_p, savemat, timestamp
from flow.misc import wordhash
try:
    from .train_factor import train
except ImportError:
    # Won't be able to train without psytrack installed, but should be able
    # to work with saved .psy files fine.
    pass


class FacTracker(object):
    """PsyTracker."""

    def __init__(self, runs, facpars=None, verbose=False, force=False):
        """Init."""
        self._runs = runs
        self._mouse = self.runs.parent

        if facpars is None:
            facpars = {}
        self._facpars = config.params()['factrack_defaults']
        self._facpars.update(facpars)
        self._facpars_word = None
        self._runs_word = None

        self._path = paths.factrack(
            self.mouse.mouse, self.facpars_word, self.runs_word)

        self._load_or_train(verbose=verbose, force=force)

        self._weight_labels = None

        self._confusion_matrix = None

    def __repr__(self):
        """Repr."""
        return "FacTracker(path={})".format(self.path)

    @property
    def mouse(self):
        """Return the Mouse object."""
        return self._mouse

    @property
    def runs(self):
        """Return the MouseRunSorter object."""
        return self._runs

    @property
    def facpars(self):
        """Return the parameters for the FacTracker."""
        return deepcopy(self._facpars)

    @property
    def path(self):
        """The path to the saved location of the data."""
        return self._path

    @property
    def data(self):
        """Return the data used to fit the model."""
        return deepcopy(self.d['data'])

    @property
    def fits(self):
        """Return the fit weights for all parameters."""
        return deepcopy(self.d['results']['model_weights'])

    @property
    def inputs(self):
        """Return the input data formatted for the model."""
        from psytrack.helper.helperFunctions import read_input
        return read_input(self.data, self.weights_dict)

    @property
    def weights_dict(self):
        """Return the dictionary of weights that were fit."""
        return self.pars['weights']

    @property
    def weight_labels(self):
        """The names of each fit weight, order matched to results.

        If a label is repeated, the first instance is the closest in time to
        the current trial, and they step back 1 trial from there.

        """
        if self._weight_labels is None:
            labels = []
            for weight in sorted(self.weights_dict.keys()):
                labels += [weight] * self.weights_dict[weight]
            self._weight_labels = labels
        return deepcopy(self._weight_labels)

    @property
    def facpars_word(self):
        """Return the hash word of the current parameters."""
        if self._pars_word is None:
            self._pars_word = wordhash.word(self.facpars, use_new=True)
        return self._pars_word

    @property
    def runs_word(self):
        """Return the hash word of the dates."""
        if self._runs_word is None:
            runs_list = [str(r) for r in self.runs]
            self._runs_word = wordhash.word(runs_list, use_new=True)
        return self._runs_word

    def predict(self, data=None):
        """Return predicted lick probability for every trial.

        Parameters
        ----------
        data : np.ndarray, optional
            If not None, the input data to make predictions from. Should be
            (ntrials x nweights), with the order matching weight_labels(). If
            a bias term was fit, the values should be all 1's.

        Returns
        -------
        prediction : np.ndarray
            A length ntrials array of values on (0, 1) that corresponds to the
            predicted probability of licking on each trial.

        """
        if data is None:
            g = self.inputs
        else:
            g = data
        X = np.sum(g.T * self.fits, axis=0)

        return 1 / (1 + np.exp(-X))

    def confusion_matrix(self):
        """Confusion matrix for the models precision in predicting lick trials.

        See also:
        https://en.wikipedia.org/wiki/Confusion_matrix

        Returns
        -------
        np.array (2x2)
            [[true negatives, false positives],
             [false negatives, true positives]]

        """
        if self._confusion_matrix is None:
            prediction = (self.predict() > 0.5)
            licked = (self.data['y'] == 2)

            TP = sum(prediction & licked)
            FP = sum(prediction & ~licked)

            TN = sum(~prediction & ~licked)
            FN = sum(~prediction & licked)
            self._confusion_matrix = np.array([[TN, FP], [FN, TP]])
            # self._confusion_matrix = sklearn.metrics.confusion_matrix(
            #     licked, prediction)

        return self._confusion_matrix

    def precision(self):
        """Precision of the model estimation.

        Fraction of all predicted lick trials where the mouse actually licked.

        """
        TN, FP, FN, TP = self.confusion_matrix().ravel()
        return TP / float(TP + FP)

    def recall(self):
        """Recall of the model estimation.

        Fraction of all real lick trials correctly predicted by the model.

        """
        TN, FP, FN, TP = self.confusion_matrix().ravel()
        return TP / float(TP + FN)

    def accuracy(self):
        """Accuracy of the model estimation.

        Fraction of trials correctly predicted.

        """
        TN, FP, FN, TP = self.confusion_matrix().ravel()
        return (TP + TN) / float(TP + FP + TN + FN)

    def f1_score(self):
        """F_1 score of the model.

        Combined precision and recall to get a single measure of model
        performance. 1 for a perfect model, 0 for completely wrong.

        See also:
        https://en.wikipedia.org/wiki/F1_score

        """
        return 2 * (self.precision() * self.recall()) / \
            (self.precision() + self.recall())

    def _load_or_train(self, verbose=False, force=False):
        if not force:
            try:
                self.d = loadmat(self.path)
                found = True
            except IOError:
                found = False
                if verbose:
                    print('No FacTracker found, re-calculating (pars_word=' +
                          self.facpars_word + ', runs_word=' + self.runs_word +
                          ').')
            else:
                # Matfiles can't store None, so they have to be converted
                # when saved to disk. I think this is the only place it
                # should be necessary.
                if verbose:
                    print('Saved FacTracker found, loading: ' + self.path)
                if 'missing_trials' in self.d['data'] and \
                        np.isnan(self.d['data']['missing_trials']):
                    self.d['data']['missing_trials'] = None

        if force or not found:
            if self.facpars['updated'] != \
                    config.params()['factrack_defaults']['updated']:
                raise ValueError(
                    'Unable to re-calculate old FacTracker version: {}'.format(
                        self.facpars['updated']))
            facpars = self.facpars
            facpars.pop('updated')
            data, results, initialization = \
                train(self.runs, verbose=verbose, **facpars)

            self.d = {
                'data': data,
                'initialization': initialization,
                'pars': self.pars,
                'results': results,
                'timestamp': timestamp()}
            mkdir_p(opath.dirname(self.path))

            yaml_path = '{}_{}'.format(
                opath.splitext(self.path)[0], 'pars.yml')
            if not opath.exists(yaml_path):
                with open(yaml_path, 'wb') as f:
                    yaml.dump(self.facpars, f, encoding='utf-8')

            savemat(self.path, matlabifypars(self.d))
