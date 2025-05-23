DB_PATH = '/mnt/d/ml_projects/va_mlops/airflow/dags/lead_scoring/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db'

FILE_PATH = '/mnt/d/ml_projects/va_mlops/airflow/dags/lead_scoring/'

TRACKING_URI = TRACKING_URI = "http://127.0.0.1:5000"

# experiment, model name and stage to load the model from mlflow model registry
MODEL_NAME = "LightGBM"
STAGE = "Production"
EXPERIMENT = "Lead_scoring_training_pipeline"

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = ['total_leads_droppped', 'referred_lead',
                            'city_tier_1.0', 'city_tier_2.0', 'city_tier_3.0',
                            'first_platform_c_Level0', 'first_platform_c_Level3',
                            'first_platform_c_Level7', 'first_platform_c_Level1',
                            'first_platform_c_Level2', 'first_platform_c_Level8',
                            'first_platform_c_others', 'first_utm_medium_c_Level0',
                            'first_utm_medium_c_Level2', 'first_utm_medium_c_Level6',
                            'first_utm_medium_c_Level3', 'first_utm_medium_c_Level4',
                            'first_utm_medium_c_Level9', 'first_utm_medium_c_Level11',
                            'first_utm_medium_c_Level5', 'first_utm_medium_c_Level8',
                            'first_utm_medium_c_Level20', 'first_utm_medium_c_Level13',
                            'first_utm_medium_c_Level30', 'first_utm_medium_c_Level33',
                            'first_utm_medium_c_Level16', 'first_utm_medium_c_Level10',
                            'first_utm_medium_c_Level15', 'first_utm_medium_c_Level26',
                            'first_utm_medium_c_Level43', 'first_utm_medium_c_others',
                            'first_utm_source_c_Level2', 'first_utm_source_c_Level0',
                            'first_utm_source_c_Level7', 'first_utm_source_c_Level4',
                            'first_utm_source_c_Level6', 'first_utm_source_c_Level16',
                            'first_utm_source_c_Level5', 'first_utm_source_c_Level14',
                            'first_utm_source_c_others']

# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = FEATURES_TO_ENCODE = ['city_tier', 'first_platform_c',
                      'first_utm_medium_c', 'first_utm_source_c']
