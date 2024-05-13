import os
import time
import logging
import logging.config

class Settings(object):
    def __init__(self):
        self.input_dir   = os.environ.get('INPUT_DIR', '.')
        self.output_dir  = os.environ.get('OUTPUT_DIR', '.')

        # DEPRECATED
        # get inputs from JSON file (useful for testing)
        self.input_file  = os.environ.get('INPUT_FILE', None)

class UTCFormatter(logging.Formatter):
    '''Convert logging timestamp to UTC'''
    converter = time.gmtime

class Logging(object):
    def __init__(self, *args, **kwargs):
        '''Configure logging format'''

        log_level        = os.environ.get('LOG_LEVEL', 'INFO')

        LOGGING = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'utc': {
                    '()': UTCFormatter,
                    'format': '[%(levelname)s] [%(module)s] - %(message)s',
                    'datefmt': '%Y-%m-%dT%H:%M:%OS'
                }
            },
            'handlers': {
                'pennsieve': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'utc',
                }
            },
            'root': {
                'handlers': ['pennsieve'],
                'level': log_level,
           }
        }

        logging.config.dictConfig(LOGGING)
        self.logger = logging.getLogger()
