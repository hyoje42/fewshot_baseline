## experimatal settings

import os
import sys
import shutil


class configuration:
    def __init__(self):

        ## base settings
        # self.data_augmentation = False
        # self.num_predictions = 20
        self.epoch_start = 0

        # parse gpu
        self.gpu = 0
        
        # model params
        self.n_examples = 600
        self.x_dim = "84,84,3"

        # training hyper-parameters
        self.n_episodes = 100 # test interval
        self.n_way = 5
        self.n_shot = 5
        self.n_query = 15 # 클래스당
        self.n_epochs = 2000
        
        # test hyper-parameters
        self.n_test_way = 5
        self.n_test_shot = 5
        self.n_test_query = 15
        self.n_test_episodes = 600

        # optimization params
        self.lr = 0.001
        self.step_size = 10000
        self.gamma = 0.5
        self.patience = 200

        # dataset params
        self.dataset = 'mini'
        self.ratio = 1.0
        self.pkl = True

        # save and restore params
        self.seed = 777
        self.exp_name = 'baseline_model'
        self.alg = "base"
        self.force = False

    def initialize(self):
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/' + self.exp_name):
            os.makedirs('results/' + self.exp_name)
        else:
            # when exp_name is overlaped
            if self.force:
                shutil.rmtree('results/' + self.exp_name)
                os.makedirs('results/' + self.exp_name)
            elif self.command == 'infer':
                print('inference from {}'.format(self.exp_name))
            else:
                print('experiment name is overlaped...')
                sys.exit(1)
            
        if not os.path.exists('results/' + self.exp_name + '/' + 'models'):
            os.makedirs('results/' + self.exp_name + '/' + 'models')
#         os.system('cp main_train.py results' + '/' + self.exp_name + '/' + 'main_train.py.backup')
#         os.system('cp models/models.py results' + '/' + self.exp_name + '/' + 'models.py.backup')
#         os.system('cp main.py results' + '/' + self.exp_name + '/' + 'main.py.backup')
        
        f = open('results/' + self.exp_name + '/log.txt', 'a')
        # print(self.__dict__, file=f)
        for key in self.__dict__.keys():
            print('{} = {}'.format(key, self.__dict__[key]), file=f)
        f.close()

        
        
    '''
        ## base settings
        self.batch_size = 32
        self.num_classes = 10
        self.epochs = 2
        self.data_augmentation = False
        self.num_predictions = 20


        # parse gpu
        self.gpu = 0
        # model params
        self.n_examples = 600
        self.x_dim = "84,84,3"
        self.h_dim = 64
        self.z_dim = 64

        # training hyper-parameters
        self.n_episodes = 100 # test interval
        self.n_way = 5
        self.n_shot = 5
        self.n_query = 15 # 클래스당
        self.n_epochs = 2100
        # test hyper-parameters
        self.n_test_way = 5
        self.n_test_shot = 5
        self.n_test_query = 15

        # optimization params
        self.lr = 0.001
        self.step_size = 10000
        self.gamma = 0.5
        self.patience = 200

        # dataset params
        self.dataset = 'mini'
        self.ratio = 1.0
        self.pkl = True

        # save and restore params
        self.seed = 1000
        self.exp_name = 'baseline_model'

'''