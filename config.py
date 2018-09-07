# RATE = '22050' 
RATE = '16000' 
CATES = [g + '_' + a for g in ['female', 'male']  for a in ['north', 'central', 'south']]
MAP = {
    'female_north': 0,
    'female_central': 1, 
    'female_south': 2,
    'male_north': 3,
    'male_central': 4,
    'male_south': 5
}
BASE_ORIGINAL_PRIVATE_TEST = './data/private_test/'
BASE_ORIGINAL_TRAIN = './data/train/'
BASE_ORIGINAL_PUBLIC_TEST = './data/public_test/'
BASE_TRAIN = './data/wav' + RATE + '/'
BASE_PUBLIC_TEST = './data/wav' + RATE + '/public_test/'
BASE_PRIVATE_TEST = './data/wav' + RATE + '/private_test/'
TRAINING_GT = './csv_data/training_groundtruth.csv'
TEST_GT = './csv_data/test_groundtruth.csv'
NUM_WORKERS = 16 

INFER_ONLY = True # change this to False to train the model again 
