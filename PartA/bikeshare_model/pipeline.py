import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer
from bikeshare_model.processing.features import WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler
from bikeshare_model.processing.features import WeekdayOneHotEncoder

bikeshare_pipe = Pipeline([
    ('weekday_imputer', WeekdayImputer(variable=config.model_config_.weekday_var)),
    ('weathersit_imputer', WeathersitImputer(variable=config.model_config_.weathersit_var)),

    ##==========Mapper======##
    ('map_yr', Mapper(variable=config.model_config_.yr_var, mappings=config.model_config_.yr_mappings)),
    ('map_mnth', Mapper(variable=config.model_config_.mnth_var, mappings=config.model_config_.mnth_mappings)),
    ('map_season', Mapper(variable=config.model_config_.season_var, mappings=config.model_config_.season_mappings)),
    ('map_weathersit', Mapper(variable=config.model_config_.weathersit_var, mappings=config.model_config_.weathersit_mappings)),
    ('map_holiday', Mapper(variable=config.model_config_.holiday_var, mappings=config.model_config_.holiday_mappings)),
    ('map_workingday', Mapper(variable=config.model_config_.workingday_var, mappings=config.model_config_.workingday_mappings)),
    ('map_hr', Mapper(variable=config.model_config_.hr_var, mappings=config.model_config_.hr_mappings)),

    ('numerical_outlier_rem', OutlierHandler(config.model_config_.numerical_features)),
    ('weekday_oneHotenc', WeekdayOneHotEncoder(variable=config.model_config_.weekday_var, encoder=OneHotEncoder(sparse_output=False))),

    # scale
    ('scaler', StandardScaler()),

    # Model fit
    ('model_rf', RandomForestRegressor(n_estimators=config.model_config_.n_estimators, 
                                       max_depth=config.model_config_.max_depth,
                                       max_features=config.model_config_.max_features,
                                       random_state=config.model_config_.random_state))
])