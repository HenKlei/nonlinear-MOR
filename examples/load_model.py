import importlib


def load_full_order_model(example, spatial_shape, num_time_steps, additional_parameters={}):
    try:
        imported_module = importlib.import_module(example)
        create_model = getattr(imported_module, 'create_model')
        fom = create_model(spatial_shape, num_time_steps, **additional_parameters)
        return fom
    except Exception as e:
        raise e


def load_landmark_function(example):
    try:
        imported_module = importlib.import_module(example)
        get_landmarks = getattr(imported_module, 'get_landmarks')
        return get_landmarks
    except Exception as e:
        raise e
