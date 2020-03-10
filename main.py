"""
2020-03-10

"""
# imports base packages
import argparse, random
import torch
from torch.optim.lr_scheduler import StepLR

# custom packages
from configs.conf import configuration
from dataset_mini import *
from utils.utils import train, evaluation, cal_num_parameters
from models import models
from models import resnet

# deal with params
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--exp_name', type=str, dest='exp_name', default='debug', help="model_name")
parser.add_argument('-g', '--gpu', type=int, dest='gpu', default=1, help="gpu")
parser.add_argument('--n_epochs', type=int, default=2000, help="epoch") # 2000
parser.add_argument('--alg', type=str, default='proto_res34', help="alg")
parser.add_argument('--command', type=str, default='train', help="train or infer")
parser.add_argument('-l', '--load', type=str, dest='load_model', default='', help='name of model')
parser.add_argument('-f', '--force', dest='force', action='store_true', help='remove by force (default : False)')
# get_args, _ = parser.parse_known_args('-g 1 --alg convex -n debug -f'.split())
get_args, _ = parser.parse_known_args()

args = configuration()
args.exp_name = get_args.exp_name
args.gpu = get_args.gpu
args.n_epochs = get_args.n_epochs
args.alg = get_args.alg
args.command = get_args.command
args.load_model = get_args.load_model
args.force = get_args.force

args.initialize()

# set environment variables: gpu, num_thread
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# RANDOM SEED
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

im_width, im_height, channels = list(map(int, args.x_dim.split(',')))
print(im_width, im_height, channels)

# Step 1: init dataloader
print("init data loader")
if args.dataset == 'mini':
    if args.command == 'train':
        loader_train = dataset_mini(args.n_examples, args.n_episodes, 'train', args)
        loader_train.load_data_pkl()
    else:
        loader_test = dataset_mini(args.n_examples, args.n_test_episodes, 'test', args)
        loader_test.load_data_pkl()
    loader_val = dataset_mini(args.n_examples, args.n_episodes, 'val', args)
    loader_val.load_data_pkl()
# elif dataset=='tiered':
#     loader_train = dataset_tiered(n_examples, n_episodes, 'train', args_data)
#     loader_val   = dataset_tiered(n_examples, n_episodes, 'val', args_data)

# Step 2: init neural networks
print("init neural networks")

# construct the model
print('alg name : ', args.alg)
print('exp name : ', args.exp_name)
if args.alg == 'cycle_1':
    model = models.PrototypicalCycle(args, optimizer=torch.optim.Adam)
elif args.alg == 'proto_1':
    model = models.Prototypical(args, optimizer=torch.optim.Adam, encoder=resnet.resnet18)
elif args.alg == 'proto_2':
    model = models.Prototypical(args, optimizer=torch.optim.Adam, encoder=models.CNNEncoder_average)
elif args.alg == 'proto_res34':
    model = models.Prototypical(args, optimizer=torch.optim.Adam, encoder=resnet.resnet34)
elif args.alg == 'cycle_res34':
    model = models.PrototypicalCycleQueryProto(args, optimizer=torch.optim.Adam, encoder=resnet.resnet34)
elif args.alg in ['cycle_2', 'cycle_3']:
    model = models.PrototypicalCycleQueryProto(args, optimizer=torch.optim.Adam)
elif args.alg == 'cycle_for':
    model = models.PrototypicalCycleQueryProto(args, optimizer=torch.optim.Adam)
elif args.alg == 'convex':
    model = models.ConvexProto(args, optimizer=torch.optim.Adam, encoder=resnet.resnet18)
else:
    print('model error')
model.cuda(0)

# calculate the number of parameters
LOG_FILE = 'results/' + args.exp_name + '/log.txt'
with open(LOG_FILE, 'a') as f:
    print(cal_num_parameters(model.parameters()), file=f)

# optimizer scheduler
model_scheduler = StepLR(model.optimizer, step_size=args.step_size, gamma=args.gamma)

if args.command == 'infer':
    print('test ...')
    if args.load_model.split('.')[-1] == '.t7':
        weight_path = args.load_model
    else:
        weight_path = './results/{:s}/models/'.format(args.load_model)
        candidate = [w for w in os.listdir(weight_path) if '.t7' in w]
        weight_path = os.path.join(weight_path, sorted(candidate)[-1])
    model.load_state_dict(torch.load(weight_path))
    print('Loading Parameters from {}'.format(weight_path))

    ## validation
    val_loss = []
    val_loss_orig = []
    val_loss_reverse = []
    val_acc = []

    ## test
    test_loss = []
    test_loss_orig = []
    test_loss_reverse = []
    test_acc = []

    if 'cycle' in args.alg:
        val_loss_orig, val_acc, val_loss_reverse = evaluation(args, 1, model, loader_val, dataset='val')
        val_loss = np.mean(val_loss_orig) + np.mean(val_loss_reverse)
        test_loss_orig, test_acc, test_loss_reverse = evaluation(args, 1, model, loader_test, dataset='test')
        test_loss = np.mean(test_loss_orig) + np.mean(test_loss_reverse)
    else:
        val_loss, val_acc = evaluation(args, 1, model, loader_val, dataset='val')
        test_loss, test_acc = evaluation(args, 1, model, loader_test, dataset='test')
    print('\nvalidation acc : {:.5f}'.format(np.mean(val_acc)))
    print('test acc : {:.5f}'.format(np.mean(test_acc)))

elif args.command == 'train':
    # Step 3: Train and validation
    print("\nTraining...")
    print('Total epoch : ', args.n_epochs)
    best_acc = 0.0
    best_loss = np.inf
    wait = 0

    for ep in range(args.epoch_start, args.n_epochs):
        if 'cycle' in args.alg:
            ## training
            tr_loss_orig, tr_acc, tr_loss_reverse = train(args, ep, model, model_scheduler, loader_train)
            tr_loss = np.mean(tr_loss_orig) + np.mean(tr_loss_reverse)
            ## validation
            val_loss_orig, val_acc, val_loss_reverse = evaluation(args, ep, model, loader_val, dataset='val')
            val_loss = np.mean(val_loss_orig) + np.mean(val_loss_reverse)
        else:
            ## training
            tr_loss, tr_acc = train(args, ep, model, model_scheduler, loader_train)
            ## validation
            val_loss, val_acc = evaluation(args, ep, model, loader_val, dataset='val')

        print('\nepoch:{}, tr_loss:{:.5f}, tr_acc:{:.5f}, val_loss:{:.5f}, val_acc:{:.5f}'
              .format(ep+1, np.mean(tr_loss), np.mean(tr_acc), np.mean(val_loss), np.mean(val_acc)))

        # Model Save and Stop Criterion
        # save : val_acc가 best 보다 잘 나왔을 때
        #        ep 100 단위로
        # break : patience 200 이상
        cond1 = (np.mean(val_acc) > best_acc)
        cond2 = (np.mean(val_loss) < best_loss)

        if cond1 or cond2:
            best_acc = np.mean(val_acc)
            best_loss = np.mean(val_loss)
            print('best val loss:{:.5f}, acc:{:.5f}'.format(best_loss, best_acc))

            # save model
            torch.save(model.state_dict(), 'results/{:s}/models/{:s}_{:0>6d}_model.t7'.format(args.exp_name, args.alg, (ep + 1) * args.n_episodes))

            f = open(LOG_FILE, 'a')
            if 'cycle' in args.alg:
                print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format((ep + 1) * args.n_episodes, best_loss,
                                                              np.mean(val_loss_orig), np.mean(val_loss_reverse),
                                                              best_acc), file=f)
            else:
                print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format((ep + 1) * args.n_episodes, np.mean(tr_loss), np.mean(tr_acc),
                                                              best_loss, best_acc), file=f)
            f.close()

            wait = 0

        else:
            wait += 1
            if ep % 100 == 0:
                torch.save(model.state_dict(), 'results/{:s}/models/{:s}_{:0>6d}_model.t7'.format(
                args.exp_name, args.alg, (ep + 1) * args.n_episodes))

                f = open(LOG_FILE, 'a')
                if 'cycle' in args.alg:
                    print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format((ep + 1) * args.n_episodes, np.mean(val_loss),
                                                                  np.mean(val_loss_orig), np.mean(val_loss_reverse),
                                                                  np.mean(val_acc)), file=f)
                else:
                    print('{} {:.5f} {:.5f} {:.5f} {:.5f}'.format((ep + 1) * args.n_episodes, np.mean(tr_loss), np.mean(tr_acc),
                                                                  np.mean(val_loss), np.mean(val_acc)), file=f)
                f.close()

        if wait > args.patience and ep > args.n_epochs:
            break
else:
    print('{} : this command is unavailable'.format(args.command))
