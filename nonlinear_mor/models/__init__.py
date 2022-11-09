from .models import AnalyticalModel, WrappedpyMORModel

from .reduced_models import ReducedSpacetimeModel


def load_model(model_class, model_dictionary):
    return model_class.load_model(**model_dictionary)


def load_model_from_file(model_class, filepath):
    with open(filepath, 'rb') as f:
        model_dictionary = pickle.load(f)
    return load_model(model_class, model_dictionary)
