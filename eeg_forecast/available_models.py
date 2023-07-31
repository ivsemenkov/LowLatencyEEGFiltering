from darts.models import DLinearModel, NHiTSModel, NLinearModel, TCNModel, TransformerModel


AVAILABLE_DARTS_MODELS = {
        'NHiTSModel': NHiTSModel,
        'TCNModel': TCNModel,
        'TransformerModel': TransformerModel,
        'DLinearModel': DLinearModel,
        'NLinearModel': NLinearModel
    }
