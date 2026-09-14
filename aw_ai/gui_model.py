"""In-memory GUI overrides; never save or change checkpoint files."""
import math


def set_value_weight(network, weight):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Neural value weight must be between 0 and 1')
    network.search_settings = dict(network.search_settings or {},
                                   mode='guided', neural_value_weight=weight)
