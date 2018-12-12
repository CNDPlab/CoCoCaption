import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('test_logger')


logger.info('start')

for i in tqdm(range(100)):
    time.sleep(1)
    logger.info('here')



