DB_PATH = '/mnt/d/ml_projects/va_mlops/airflow/dags/lead_scoring/'
DB_FILE_NAME = 'lead_scoring_data_cleaning.db'
DATA_DIRECTORY = '/mnt/d/ml_projects/va_mlops/lead_scoring/_data_pipeline/data/'
INTERACTION_MAPPING = '/mnt/d/ml_projects/va_mlops/lead_scoring/_data_pipeline/mapping/interaction_mapping.csv'


INDEX_COLUMNS_TRAINING = ['created_date', 'city_tier', 'first_platform_c',
                'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
                'referred_lead', 'app_complete_flag']

INDEX_COLUMNS_INFERENCE = ['created_date', 'city_tier', 'first_platform_c',
                'first_utm_medium_c', 'first_utm_source_c', 'total_leads_droppped',
                'referred_lead']

NOT_FEATURES = ['created_date', 'assistance_interaction', 'career_interaction',
                'payment_interaction', 'social_interaction', 'syllabus_interaction']


