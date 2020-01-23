from datascience.ml.neural.checkpoints import create_model
from datascience.ml.neural.models import InceptionQRouting
from datascience.ml.neural.reinforcement import fit
from datascience.ml.neural.reinforcement.game import OffshoreRegatta

model = create_model(model_class=InceptionQRouting)

fit(
    model, OffshoreRegatta, game_params={'source': 'grib_gfs_2018'},
    training_params=None, predict_params=None, validation_params=None,
    export_params=None, optim_params=None
)

# def run(xp_name, **kwargs):
#    setup_and_run_experiment(xp_name, OffshoreRegatta, model_class=Net, loss=MSELoss(), **kwargs)
