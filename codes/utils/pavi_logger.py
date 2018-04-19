import logging
import os
import sys
import time
from datetime import datetime
from getpass import getuser
from socket import gethostname
from threading import Thread

import requests
from six.moves.queue import Empty, Queue

class PaviLogger(object):
    def __init__(self, url, username, password=None, instance_id=None, timeout=5):
        self.timeout = timeout
        self.url = url
        self.username = username
        if password is not None:
            self.password = str(password)
        else:
            password = os.getenv('PAVI_PASSWORD')
            if password:
                self.password = password
            else:
                raise ValueError('Pavi password not specified')
        self.instance_id = instance_id
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    def setup(self, info):
        logging.info('connecting pavi service ' + self.url)
        post_data = dict(
            time=str(datetime.now()),
            username=self.username,
            password=self.password,
            instance_id=self.instance_id,
            session_file=info['session_file'], # main_file?
            session_text=info['session_text'], # opt string
            model=info['model_name'],
            model_text=info['model_text'], # NO
            work_dir=info['work_dir'], # model dir
            device='{}@{}'.format(getuser(), gethostname()))
        try:
            response = requests.post(self.url, json=post_data, timeout=self.timeout)
        except Exception as ex:
            logging.error('fail to connect to pavi service: %s', ex)
        else:
            if response.status_code == 200:
                self.instance_id = response.content.decode("utf-8")
                logging.info('pavi service connected, instance_id: %s',
                             self.instance_id)
                self.log_queue = Queue()
                self.log_thread = Thread(target=self.post_log, args=(3, 3, 3))
                self.log_thread.daemon = True
                self.log_thread.start()
                return True
            else:
                logging.error('fail to connect to pavi service, status code: '
                              '%d, err message: %s', response.status_code,
                              response.reason)
        return False

    def log(self, logs):
        if hasattr(self, 'log_queue'):
            self.log_queue.put(logs.copy())

    def post_log(self, max_retry, queue_timeout, req_timeout):
        while True:
            try:
                log = self.log_queue.get(timeout=queue_timeout)
            except Empty:
                time.sleep(1)
            except Exception as ex:
                logging.error('fail to get logs from queue: %s', ex)
            else:
                
                log['time'] = str(log['time'])
                log['instance_id'] = self.instance_id
                log['msg'] = ''  # reserved for future use (eg. error message)
                retry = 0
                
                
                while retry < max_retry:
                    try:
                        response = requests.post(
                            self.url, json=log, timeout=req_timeout)
                    except Exception as ex:
                        retry += 1
                        logging.warn('error when posting logs to pavi: %s', ex)
                    else:
                        status_code = response.status_code
                        if status_code == 200:
                            
                            break
                        else:
                            logging.warn(
                                'unexpected status code: %d, err msg: %s',
                                status_code, response.reason)
                            retry += 1
                if retry == max_retry:
                    logging.error('fail to send logs of iteration %d',
                                  log['iter_num'])

if __name__ == "__main__":
    # for test
    url = 'http://pavi.parrotsdnn.org/log'
    username = 'testtest'
    password = '123456'

    logger = PaviLogger(url, username, password=password)

    info = {}
    info['session_file'] = 'session file'
    info['session_text'] = 'session text'
    info['model_name'] = 'test_test'
    info['model_text'] = 'model text'
    info['work_dir'] = 'model dir'

    logger.setup(info)

    for i in range(1,5000):
        train_loss = 10+i/100
        train_PSNR = 20+i/200
        send_data_train = {'loss':train_loss, 'acc_PSNR':train_PSNR}
        log_data = {'time':str(datetime.now()), 'flow_id':'train', 'iter_num':i, 'outputs':send_data_train}
        logger.log(log_data)
        time.sleep(0.1)
        if i%100 == 0:
            test_loss = 30+i/100
            test_PSNR = 40+i/200
            send_data_test = {'loss':test_loss, 'acc_PSNR':test_PSNR}
            log_data = {'time':str(datetime.now()), 'flow_id':'test', 'iter_num':i, 'outputs':send_data_test}
            logger.log(log_data)