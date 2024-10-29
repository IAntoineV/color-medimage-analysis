

from sklearn.mixture import GaussianMixture





class WEstimations():



    @staticmethod
    def GMM_based_estimation(data, nb_components, **kwargs):

        model = GaussianMixture(n_components=nb_components, **kwargs)

        model.fit(data)

        return model.means_, model