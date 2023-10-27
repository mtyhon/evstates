import torch, sklearn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, re, glob, h5py, matplotlib, shutil, math
import torch.utils.data as utils
import scipy

print('Scipy Version: ', scipy.__version__)

print(torch.__version__)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions import Categorical
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, norm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, brier_score_loss, \
    auc, roc_curve, log_loss, confusion_matrix, precision_recall_curve, average_precision_score
from scipy.misc import imresize
from triplet_utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector # Strategies for selecting triplets within a minibatch
from torch.utils.data.sampler import BatchSampler
from scipy import signal, stats
import math

from BBBlayers import BBBConv2d, BBBConv1d, BBBLinearFactorial, FlattenLayer

test_KIC_file = '/home/z3384751/PycharmProjects/AstroBlasto/TestKIC(GOTY).txt'

ratio_data = pd.read_csv('/home/z3384751/AstroBlasto/Population_Ratio.dat', header=0, delim_whitespace=True)
ratio_index = ratio_data['val'].values
ratio_dnu = ratio_data['dnu'].values
ratio_heb_ratio = ratio_data['heb/rgb'].values
ratio_rgb_ratio = ratio_data['rgb/heb'].values
heb_fraction_weight = ratio_rgb_ratio/(ratio_heb_ratio+ratio_rgb_ratio)
rgb_fraction_weight = 1-heb_fraction_weight
print('HeB Fraction Weight: ', heb_fraction_weight)
print('Ratio Dnu: ', ratio_dnu)
class npz_generator(data.Dataset):
    def __init__(self, root, dim=(2000,), extension='.npz', select_kic=np.array([]),external_label=None, obs_len=82, do_perturb=True, external_kic = None):
        self.root = root  # root folder containing subfolders
        self.extension = extension  # file extension
        self.filenames = []
        self.external_label = external_label # is all Elsworth labels
        self.dim = dim  # image/2D array dimensions
        self.perturb_factor = np.sqrt(365. * 4. / obs_len)
        self.do_perturb = do_perturb
        self.external_kic = external_kic # is synced with Elsworth labels
        self.kic = []
        self.label = []

        for dirpath, dirnames, filenames in os.walk(root):
            for file in filenames:
                if file.endswith(extension):
                    file_kicz = int(file.split('-')[0])
                    if file_kicz in select_kic:
                        self.filenames.append(os.path.join(dirpath, file))
                        self.kic.append(file_kicz)
                        if self.external_label is not None:
                            self.label.append(self.external_label[np.where(self.external_kic == file_kicz)[0]])

        self.kic = np.array(self.kic)
        self.filenames = np.array(self.filenames)
        if self.external_label is not None:
            self.label = np.array(self.label)

        self.indexes = np.arange(len(self.kic))
        #self.indexes = self.indexes[np.in1d(self.kic, select_kic)]

        print('Number of files: ', len(self.filenames))
        print('Number of unique IDs: ', len(select_kic))

    def __len__(self):
        'Total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        # Get a list of filenames of the batch
        batch_filenames = self.filenames[index]
        batch_kic = self.kic[index]

        X, y, dnu_vec, flag_vec = self.__data_generation(batch_filenames)
        if self.external_label is not None:
            batch_label = self.label[index]
            return X, batch_label, dnu_vec, flag_vec, batch_kic
        else:
            return X, y, dnu_vec, flag_vec, batch_kic

    def __supergausswindow(self, interval):
        window = signal.general_gaussian(2000, p=3, sig=650)
        window = window / np.max(window)
        return np.multiply(interval, window)

    def __echelle(self, lower_limit_index, upper_limit_index, freq, dnu, power):
        modulo = []
        for i in range(lower_limit_index, upper_limit_index):
            modulo.append((np.remainder(freq[i], dnu) / dnu)[0])

        return modulo, power[lower_limit_index: upper_limit_index]

    def __create_folded_spectra(self, freq, power, dnu, dnu_err, numax, numax_err):
        if self.do_perturb:
            dnu_perturb = np.random.normal(loc=dnu, scale=dnu_err * self.perturb_factor)
            numax_perturb = np.random.normal(loc=numax, scale=numax_err * self.perturb_factor)
        else:
            dnu_perturb = dnu
            numax_perturb = numax
        lower_limit_index = np.argmin(np.abs(freq - (numax_perturb - 2 * dnu_perturb)))
        upper_limit_index = np.argmin(np.abs(freq - (numax_perturb + 2 * dnu_perturb)))
        if not lower_limit_index:
            lower_limit_index = 0
        if not upper_limit_index:
            upper_limit_index = len(freq) - 1

        eps, mod_power = self.__echelle(lower_limit_index, upper_limit_index, freq, dnu_perturb, power)
        mod_power = mod_power[np.argsort(eps)]
        eps = np.sort(eps)
        if mod_power.shape[0] == 0:
            return np.zeros(self.dim), [0.]
        reshape_pow = mod_power.reshape((len(mod_power), 1))

        resize = imresize(reshape_pow, size=(1000, 1), interp='lanczos')
        resize = np.append(resize, resize)
        resize = self.__supergausswindow(resize)
        assert resize.shape == self.dim, 'Folded Spectra is of Wrong Shape!'

        if np.max(resize, axis=0) == 0:
            return np.zeros(self.dim), [0.]
        resize /= np.max(resize, axis=0)
        return resize, dnu_perturb

    def __data_generation(self, batch_filenames):

        data = np.load(batch_filenames)
        freq = data['freq']
        power = data['power']
        numax = data['numax']
        numax_err = data['numax_err']
        dnu = data['dnu']
        dnu_err = data['dnu_err']
        y = data['pop']
        flag_vec = 0

        folded_spectra, dnu_perturb = self.__create_folded_spectra(freq, power, dnu, dnu_err, numax, numax_err)
        if (dnu_perturb == 0):  # focus training on overlapping region
            flag_vec = 1
        elif dnu > 9.:
            flag_vec = 2

        X = folded_spectra
        dnu_perturb = np.round(dnu_perturb, 2)
        return X, y, dnu_perturb, flag_vec


class OneD_Stack_Bayesian(nn.Module):
    def __init__(self, kernel_size, feature_maps, padding):
        super(OneD_Stack_Bayesian, self).__init__()
        self.kernel_size = kernel_size
        self.feature_maps = feature_maps
        self.padding = padding

        self.conv1 = BBBConv1d(1, self.feature_maps ** 1, kernel_size=self.kernel_size,
                               padding=self.padding)  # same padding 2P = K-1
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = BBBConv1d(self.feature_maps ** 1, self.feature_maps ** 2, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = BBBConv1d(self.feature_maps ** 2, self.feature_maps ** 3, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool1d(2, 2)
        self.flatten = FlattenLayer(250 * 8)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2, self.conv3, self.soft3,
                  self.pool3,
                  self.flatten]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        return x, kl

class OneD_Stack(nn.Module):
    def __init__(self, kernel_size, feature_maps, padding):
        super(OneD_Stack, self).__init__()
        self.kernel_size = kernel_size
        self.feature_maps = feature_maps
        self.padding = padding

        self.conv1 = nn.Conv1d(1, self.feature_maps ** 1, kernel_size=self.kernel_size,
                               padding=self.padding)  # same padding 2P = K-1
        self.pool1 = nn.MaxPool1d(2, 2)

        self.conv2 = nn.Conv1d(self.feature_maps ** 1, self.feature_maps ** 2, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(self.feature_maps ** 2, self.feature_maps ** 3, kernel_size=self.kernel_size,
                               padding=self.padding)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.flatten = FlattenLayer(250 * 8)

        layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3,self.pool3, self.flatten]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Simple_Linear_Layers(nn.Module):
    def __init__(self):
        super(Simple_Linear_Layers, self).__init__()
        self.fc1 = nn.Linear(4000, 512)
        self.soft1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 64)
        self.soft2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, input_dnu):
        x= self.fc1(x)
        x = self.soft1(x)
        x = self.fc2(x)
        x = self.soft2(x)
        x = torch.add(x, input_dnu.view(-1, 1))
        x= self.fc3(x)

        return x

class Simple_Classifier(nn.Module):

    def __init__(self):
        super(Simple_Classifier, self).__init__()
        self.conv_stack1 = OneD_Stack(kernel_size=31, feature_maps=2, padding=15)
        self.conv_stack2 = OneD_Stack(kernel_size=31, feature_maps=2, padding=15)
        self.linear = Simple_Linear_Layers()
        layers = [self.conv_stack1, self.conv_stack2, self.linear]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, input_dnu):  # (N,Cin,L)
        x = x.unsqueeze(1)

        stack1 = self.conv_stack1(x)
        stack2 = self.conv_stack2(x)
        concat = torch.cat((stack1.view(x.size()[0], -1), stack2.view(x.size()[0], -1)), 1)
        logits = self.linear(concat, input_dnu)
        return logits

    def get_embedding(self, x):
        return self.forward(x)



class Linear_Layers(nn.Module):
    def __init__(self):
        super(Linear_Layers, self).__init__()
        self.fc1 = BBBLinearFactorial(4000, 512)
        self.soft1 = nn.Softplus()
        self.fc2 = BBBLinearFactorial(512, 64)
        self.soft2 = nn.Softplus()
        self.fc3 = BBBLinearFactorial(64, 2)

    def forward(self, x, input_dnu):
        kl = 0
        x, _kl1 = self.fc1.fcprobforward(x)
        x = self.soft1(x)
        x, _kl2 = self.fc2.fcprobforward(x)
        x = self.soft2(x)
        x = torch.add(x, input_dnu.view(-1, 1))
        x, _kl3 = self.fc3.fcprobforward(x)
        kl = kl + _kl1 + _kl2 + _kl3

        return x, kl



class Bayes_Classifier(nn.Module):
    '''The architecture of SLOSH with Bayesian Layers'''

    def __init__(self):
        super(Bayes_Classifier, self).__init__()
        self.conv_stack1 = OneD_Stack_Bayesian(kernel_size=31, feature_maps=2, padding=15)
        self.conv_stack2 = OneD_Stack_Bayesian(kernel_size=31, feature_maps=2, padding=15)
        self.linear = Linear_Layers()
        layers = [self.conv_stack1, self.conv_stack2, self.linear]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x, input_dnu):  # (N,Cin,L)
        'Forward pass with Bayesian weights'
        x = x.unsqueeze(1)
        kl = 0

        stack1, _kl1 = self.conv_stack1(x)
        stack2, _kl2 = self.conv_stack2(x)
        concat = torch.cat((stack1.view(x.size()[0], -1), stack2.view(x.size()[0], -1)), 1)
        logits, _kl_fc = self.linear(concat, input_dnu)
        kl = kl + _kl1 + _kl2 + _kl_fc
        return logits, kl

    def get_embedding(self, x, input_dnu):
        return self.probforward(x, input_dnu)



def get_beta(m, beta_type='standard', batch_idx=None):  # m is the number of minibatches
    if beta_type == "Blundell":  # Weight Uncertainty in Neural Networks, Blundell et al. (2015), Section 3.4 pg 5:
        ###
        # The first few minibatches are heavily influenced by the complexity cost (KL), whilst the later minibatches are largely
        # influenced by the data. At the beginning of learning this is particularly useful as for the first few minibatches
        # changes in the weights due to the data are slight and as more data are seen,
        # data become more influential and the prior less influential
        ###
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    else:  # standard from Graves et al. (2011), weights KL divergence in each minibatch equally
        beta = 1 / m
    return beta


def elbo(out, y, kl, beta):
    loss = F.cross_entropy(out, y, reduction='none')
    return loss + beta * kl


def elbo_focal(out, y, kl, beta, alpha):
    target = y.view(-1,1)
    logpt = F.log_softmax(out, dim=1)
    try:
        alpha = torch.from_numpy(alpha).float().cuda().view(-1,1)
        alpha = torch.cat((1-alpha, alpha), dim=1)
    except:
        alpha = torch.Tensor([1-alpha, alpha]).float().cuda()

    #print('logpt before: ', logpt)
    logpt = logpt.gather(1,target)
    #print('logpt after: ', logpt)
    logpt = logpt.view(-1)
    pt = logpt.exp()
    at = alpha.gather(1, target)
    #print('pt: ', pt)
    logpt = logpt*at
    lowpt = (pt < 0.2).float()
    highpt = (pt >= 0.2).float() # set thresholds
    gamma = 5*lowpt + 3*highpt
    #print('gamma: ', gamma)
    loss = -1*(1-pt)**gamma * logpt
    return loss + beta * kl


def train(model, model_optimizer, input_image, input_label, input_dnu, batch_idx, train_dataloader, second_rc_factor = 1.): # upweight for Kepler (5), but downweight for K2 (0.25)? If using focal loss with alpha, set this to 1.

    digitized_dnu = np.digitize(input_dnu, bins=np.linspace(2.0, 9.2, num=37)).squeeze()
    digitized_dnu = np.clip(digitized_dnu, a_max=35, a_min=None)
    alpha = heb_fraction_weight[digitized_dnu]

    model_optimizer.zero_grad()

    # Combined forward pass
    outputs, kl = model.probforward(input_image, input_dnu)  # Bayesian

    #outputs = model(input_image, input_dnu)  # Non-Bayesian
 

    # Calculate loss and backpropagate
    loss = elbo(outputs, input_label, kl, get_beta(math.ceil(len(train_dataloader) / 32), beta_type="Blundell", batch_idx=batch_idx))  # 32 is batch size
    #loss = elbo_focal(outputs, input_label, kl=0, beta=0, alpha=alpha)  # 32 is batch size
    #loss  = F.cross_entropy(outputs, input_label, reduction='none')


    if loss[input_dnu.squeeze(1) > 9].size()[0] < 1:
        loss = torch.mean(loss[input_dnu.squeeze(1) <= 9])
    elif loss[input_dnu.squeeze(1) <= 9].size()[0] < 1:
        loss = second_rc_factor*torch.mean(loss[input_dnu.squeeze(1) > 9])
    else:
        loss = torch.mean(loss[input_dnu.squeeze(1) <= 9]) + second_rc_factor*torch.mean(loss[input_dnu.squeeze(1) > 9])

    loss = torch.mean(loss)
    loss.backward()


    # Update parameters
    model_optimizer.step()
    pred = torch.max(outputs, dim=1)[1]
    correct = torch.sum(pred.eq(input_label)).item()
    total = input_label.numel()
    acc = 100. * correct / total
    return loss.item(), acc, pred


def validate(model, val_dataloader):
    model.eval()  # set to evaluate mode
    val_loss = 0
    val_batch_acc = 0
    val_batches = 0
    val_pred = []
    val_truth = []
    val_prob = []
    val_kic_array = []
    val_dnuz = []
    val_logits = []
    for batch_idy, val_data in enumerate(val_dataloader, 0):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age

        val_image = val_data[0].cuda().float()
        val_label = val_data[1].cuda().long().squeeze(1)
        val_dnu_var = val_data[2].cuda().float()
        val_flag = val_data[3].cuda().float()
        val_kic = val_data[4].cuda().float()

        val_image = val_image[val_flag != 1]
        val_label = val_label[val_flag != 1]
        val_dnu_var = val_dnu_var[val_flag != 1]
        val_kic = val_kic[val_flag != 1]

        if len(val_label) < 1:
            continue

        digitized_dnu = np.digitize(val_dnu_var, bins=np.linspace(2.0, 9.2, num=37)).squeeze()
        digitized_dnu = np.clip(digitized_dnu, a_max=35, a_min=None)
        alpha = heb_fraction_weight[digitized_dnu]

        with torch.no_grad():
            outputs, kl = model.probforward(val_image, val_dnu_var) # Bayesian
            #outputs = model(val_image, val_dnu_var) # non-Bayesian
            val_batch_loss = elbo(outputs, val_label, kl,
                                  get_beta(math.ceil(len(val_dataloader)) / 32, beta_type="Blundell",
                                           batch_idx=batch_idy))
            #val_batch_loss = elbo_focal(outputs, val_label, kl=0,beta=0, alpha=alpha)
            #val_batch_loss = F.cross_entropy(outputs, val_label, reduction='none')
            val_logits.append(outputs.data.cpu().numpy())
        pred = torch.max(outputs, dim=1)[1]
        correct = torch.sum(pred.eq(val_label)).item()
        total = val_label.numel()
        val_loss += val_batch_loss.mean().item()
        val_batch_acc += 100. * correct / total
        val_batches += 1
        val_pred.append(pred.data.cpu().numpy())
        val_dnuz.append(val_dnu_var.data.cpu().numpy())
        val_truth.append(val_label.data.cpu().numpy())
        val_prob.append(F.softmax(outputs, dim=1).data.cpu().numpy())
        val_kic_array.append(val_kic.data.cpu().numpy())
    val_kic_array = np.concatenate(val_kic_array, 0)
    val_dnuz = np.concatenate(val_dnuz, 0)
    val_logits = np.concatenate(val_logits, 0)
    print('KIC Through Validation Pass: ', len(np.unique(val_kic_array)))

    print('Total Val KIC in Ambiguous Region: ', np.sum((val_dnuz < 9), 0))

    return (val_loss / val_batches), (val_batch_acc / val_batches), np.concatenate(val_pred, axis=0), np.concatenate(
        val_truth, axis=0), np.concatenate(val_prob, axis=0), val_kic_array, val_logits


def train_model():
    model = Bayes_Classifier()
    model.cuda()
    torch.backends.cudnn.benchmark = True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)

    folder_kic = []
    root_folder = '/data/marc/AstroBlasto/npz_BGCorr/BGCorrSpectra(365days-Consensus)/'
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filex in filenames:
            folder_kic.append(int(filex.split('-')[0]))

    folder_kic = np.unique(folder_kic)

    # train_kics, test_kics = folder_kic[~np.in1d(folder_kic, leave_out_kic)], folder_kic[np.in1d(folder_kic, leave_out_kic)]
    train_kics, test_kics = train_test_split(folder_kic, test_size=0.15, random_state=137)

    print('Number of Test KICs: ', len(test_kics))
    train_kics, val_kics = train_test_split(train_kics, test_size=0.175, random_state=137)
    print('Number of Train KICs: ', len(train_kics))
    #print('Number of Validation KICs: ', len(val_kics))
    train_gen = npz_generator(root=root_folder, select_kic=train_kics, obs_len=365)
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=5)

    val_gen = npz_generator(root=root_folder, select_kic=val_kics, obs_len=365)
    val_dataloader = utils.DataLoader(val_gen, shuffle=False, batch_size=32, num_workers=5)

    jie_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
    jie_kic = jie_data['KICID'].values
    jie_dnu = jie_data['dnu'].values

    label_data = pd.read_csv('/data/marc/AstroBlasto/Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    label_kic = label_data['KIC'].values
    label_truth = label_data['Label'].values
    train_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in train_kics])
    test_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in test_kics])
    train_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in train_kics])
    #val_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in val_kics])
    test_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in test_kics])

    plt.hist(train_dnu[train_label == 0], color='r', bins=50, alpha=0.5)
    plt.hist(train_dnu[train_label == 1], color='b', bins=25, alpha=0.5)
    print('Nb Test Dnu smaller than 9: ', np.sum(test_dnu < 9))
    print('Nb Train Dnu smaller than 9: ', np.sum(train_dnu < 9))
    print('Minimum Dnu For HeB: ', np.min(train_dnu[train_label == 1]))
    plt.yscale('log')
    plt.show()

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=20, verbose=True, min_lr=1E-6)

    n_epochs = 250
    best_loss = 1.e9
    model_checkpoint = True
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_batches = 0
        acc_cum = 0
        train_kic_array = []
        dnu_array = []
        pred_array = []
        model.train()  # set to training mode

        for batch_idx, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):

            image = data[0].cuda().float()
            label = data[1].cuda().long().squeeze(1)
            dnu = data[2].cuda().float()
            flag = data[3].cuda().float()
            train_kic = data[4].cuda().float()

            image = image[flag != 1] # flag 1 is bad stuff, flag 2 is dnu > 9
            label = label[flag != 1]
            dnu = dnu[flag != 1]
            train_kic = train_kic[flag != 1]

            if len(label) < 2:
                print('Insufficient Batch!')
                continue

            loss, acc, predz = train(model, model_optimizer, image, label, dnu, batch_idx, train_dataloader, second_rc_factor = 0.25)
            train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_batches += 1
            acc_cum += acc
            train_kic_array.append(train_kic.data.cpu().numpy())
            pred_array.append(predz.data.cpu().numpy())
            dnu_array.append(dnu.data.cpu().numpy())
        train_loss = train_loss / train_batches
        train_acc = acc_cum / train_batches
        train_kic_array = np.concatenate(train_kic_array, 0)
        pred_array = np.concatenate(pred_array, axis = 0)
        dnu_array = np.concatenate(dnu_array, axis = 0)
        print('KICs through Training Pass:', len(np.unique(train_kic_array)))
        print('Total Train KIC in Ambiguous Region: ', np.sum(train_dnu < 9))

        val_loss, val_acc, val_pred, val_truth, val_prob, val_kic, val_logits = validate(model, val_dataloader)
        scheduler.step(train_loss)  # reduce LR on loss plateau

        if epoch % 230 == 1:
            plt.hist(dnu_array[pred_array == 0], color='r', bins=50, alpha=0.5)
            plt.hist(dnu_array[pred_array == 1], color='b', bins=25, alpha=0.5)
            plt.yscale('log')
            plt.show()
            plt.close()

        print('\n\nTrain Loss: ', train_loss)
        print('Train Acc: ', train_acc)

        #val_loss = train_loss
        #val_acc = train_acc
        print('Val Loss: ', val_loss)
        print('Val Acc: ', val_acc)


        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])

        if model_checkpoint:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best or (epoch == 1000):
                filepath = '/home/z3384751/AstroBlasto/saved_models/OTF_Pytorch/365d_Downweight_LLRGB_ConsensusNoVal_Epoch%d_LLRGB_ACC:%.2f-Loss:%.3f' % (
                    epoch,val_acc, val_loss)
                print('Model saved to %s' % filepath)
                torch.save(model.state_dict(), filepath)
                # to load models:
                # the_model = TheModelClass(*args, **kwargs)
                # the_model.load_state_dict(torch.load(PATH))
            else:
                print('No improvement over the best of %.4f' % best_loss)


def train_model_k_fold():
    folder_kic = []
    root_folder = '/data/marc/AstroBlasto/npz_BGCorr/BGCorrSpectra(82days-Consensus)/'
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filex in filenames:
            folder_kic.append(int(filex.split('-')[0]))

    folder_kic = np.unique(folder_kic)

    train_kics, test_kics = train_test_split(folder_kic, test_size=0.15, random_state=137)

    print('Number of Test KICs: ', len(test_kics))
    print('Number of Train KICs: ', len(train_kics))

    label_data = pd.read_csv('/data/marc/AstroBlasto/Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    label_kic = label_data['KIC'].values
    label_truth = label_data['Label'].values
    train_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in train_kics])
    test_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in test_kics])

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=137)
    prob_over_folds = []
    truth_over_folds = []
    acc_over_folds = []
    rgb_precision_over_folds = []
    heb_precision_over_folds = []
    rgb_recall_over_folds = []
    heb_recall_over_folds = []
    fold_count = 0

    for train_idx, valid_idx in cv.split(train_kics, train_label):
        not_reset = True
        fold_count += 1
        print('Validation Fold: ', fold_count)
        #if fold_count != 1:
        #    continue
        while not_reset:

            model = Simple_Classifier()
            model.cuda()
            torch.backends.cudnn.benchmark = True
            print('train_idx: ', train_idx)
            train_gen = npz_generator(root=root_folder, select_kic=train_kics[train_idx], obs_len=82, do_perturb=True)
            train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=15)

            val_gen = npz_generator(root=root_folder, select_kic=train_kics[valid_idx], obs_len=82, do_perturb=True)
            val_dataloader = utils.DataLoader(val_gen, shuffle=False, batch_size=32, num_workers=8)

            learning_rate = 0.001
            model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1E-6)

            n_epochs = 100
            for epoch in range(1, n_epochs + 1):
                print('---------------------')
                print('Epoch: ', epoch)
                print('Validation Fold: ', fold_count)
                train_loss = 0
                train_batches = 0
                acc_cum = 0

                model.train()  # set to training mode

                for batch_idx, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'):

                    image = data[0].cuda().float()
                    label = data[1].cuda().long().squeeze(1)
                    dnu = data[2].cuda().float()
                    flag = data[3].cuda().float()

                    image = image[flag != 1]
                    label = label[flag != 1]
                    dnu = dnu[flag != 1]

                    if len(label) < 2:
                        print('Insufficient Batch!')
                        continue

                    loss, acc, predz = train(model, model_optimizer, image, label, dnu, batch_idx, train_dataloader)
                    train_loss += loss  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
                    train_batches += 1
                    acc_cum += acc

                train_loss = train_loss / train_batches
                train_acc = acc_cum / train_batches

                val_loss, val_acc, val_pred, val_truth, val_prob, val_kic, val_logits = validate(model, val_dataloader)
                scheduler.step(train_loss)  # reduce LR on loss plateau

                print('\n\nTrain Loss: ', train_loss)
                print('Train Acc: ', train_acc)

                print('Val Loss: ', val_loss)
                print('Val Acc: ', val_acc)

                if (val_acc < 60) and (epoch > 10):
                    break

                for param_group in model_optimizer.param_groups:
                    print('Current Learning Rate: ', param_group['lr'])

            not_reset = False # stop it looping again
        # After training, one last validation pass after training
        _, final_acc, final_pred, final_truth, final_prob, final_kic, final_logits = validate(model, val_dataloader)
        fold_precision_heb = precision_score(y_true=final_truth, y_pred=final_pred, pos_label=1)
        fold_precision_rgb = precision_score(y_true=final_truth, y_pred=final_pred, pos_label=0)
        fold_recall_heb = recall_score(y_true=final_truth, y_pred=final_pred, pos_label=1)
        fold_recall_rgb = recall_score(y_true=final_truth, y_pred=final_pred, pos_label=0)
        np.savez_compressed('/data/marc/AstroBlasto/82d_Simple_Classifier_NLL_Validation_Pred_Truth_Fold-%d' % fold_count, prob=final_prob, truth=final_truth,
                            kic=final_kic, logits=final_logits)

        print('Accuracy for this fold: ', final_acc)
        print('HeB Precision for this fold: ', fold_precision_heb)
        print('HeB Recall for this fold: ', fold_recall_heb)
        print('RGB Precision for this fold: ', fold_precision_rgb)
        print('RGB Recall for this fold: ', fold_recall_rgb)

        prob_over_folds.append(final_prob)
        truth_over_folds.append(final_truth)
        acc_over_folds.append(final_acc)
        print('Accumulated Accuracy Thus Far: ', acc_over_folds)
        rgb_precision_over_folds.append(fold_precision_rgb)
        rgb_recall_over_folds.append(fold_recall_rgb)
        heb_precision_over_folds.append(fold_precision_heb)
        heb_recall_over_folds.append(fold_recall_heb)

    prob_over_folds = np.concatenate(prob_over_folds, axis=0)
    truth_over_folds = np.concatenate(truth_over_folds, axis=0)

    print(prob_over_folds.shape)
    print(truth_over_folds.shape)

    print('Validation Accuracy: %.3f +/- %.3f' % (np.mean(acc_over_folds), np.std(acc_over_folds)))
    print('Validation HeB Precision: %.3f +/- %.3f' % (
    np.mean(heb_precision_over_folds), np.std(heb_precision_over_folds)))
    print('Validation HeB Recall: %.3f +/- %.3f' % (np.mean(heb_recall_over_folds), np.std(heb_recall_over_folds)))
    print('Validation RGB Precision: %.3f +/- %.3f' % (
    np.mean(rgb_precision_over_folds), np.std(rgb_precision_over_folds)))
    print('Validation RGB Recall: %.3f +/- %.3f' % (np.mean(rgb_recall_over_folds), np.std(rgb_recall_over_folds)))


def test_model(test_single=False):
    saved_model_dict = '/home/z3384751/AstroBlasto/saved_models/OTF_Pytorch/82d_ACC:94.03-Loss:63.665'
    trained_model = Bayes_Classifier()
    trained_model.load_state_dict(torch.load(saved_model_dict))
    trained_model.cuda()
    trained_model.eval()
    torch.backends.cudnn.benchmark = True

    test_pred = []
    test_truth = []
    test_mean = []
    test_dnu_vec = []
    test_aleatoric = []
    test_epistemic = []

    test_file = '/data/marc/AstroBlasto/Single_Perturb_Datasets/JieCollapsedSpectra(82daysCopy)RealisticDnu.csv'
    if test_single:
        test_KIC_file = '/home/z3384751/AstroBlasto/TestKIC(GOTY).txt'
        test_data = pd.read_table(test_KIC_file, header=None)
        leave_out_kic = test_data.iloc[:].values
        print('Number of Leave Out KIC in catalog: ', len(leave_out_kic))

        print('Testing Single Perturb Dataset...')
        test_data = pd.read_csv(test_file, header=None)
        X = test_data.iloc[:, 0:2000].values.astype(float)
        kic = test_data.iloc[:, 2000].values
        Y = test_data.iloc[:, 2001].values
        dnu = test_data.iloc[:, 2002].values  # use this for non 4-year sets
        row_max = np.max(X, axis=1)
        for i in range(len(X)):
            if row_max[i] == 0:
                print('Zero entry detected!')  # Data checking
            X[i] = X[i] / row_max[i]  # normalizing each row

        X_test = X[np.in1d(kic, leave_out_kic)]
        kic_test = kic[np.in1d(kic, leave_out_kic)]
        dnu_test = dnu[np.in1d(kic, leave_out_kic)]
        Y_test = Y[np.in1d(kic, leave_out_kic)]

        testX = torch.stack([torch.Tensor(i) for i in X_test])  # convert to torch Tensor
        testY = torch.stack([torch.Tensor(i) for i in [Y_test]]).permute(1, 0)  # convert to torch Tensor
        testdnu = torch.stack([torch.Tensor(i) for i in [dnu_test]]).permute(1, 0)  # convert to torch Tensor
        testflag = torch.zeros(testY.size()).squeeze()

        test_dataset = utils.TensorDataset(testX, testY, testdnu, testflag)
        test_dataloader = utils.DataLoader(test_dataset, shuffle=True, batch_size=32)

    else:
        folder_kic = []
        root_folder = '/data/marc/AstroBlasto/npz_BGCorr/BGCorrSpectra(82days-Consensus)/'
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filex in filenames:
                folder_kic.append(int(filex.split('-')[0]))

        folder_kic = np.unique(folder_kic)
        train_kics, test_kics = train_test_split(folder_kic, test_size=0.15, random_state=137)

        test_gen = npz_generator(root=root_folder, select_kic=test_kics, obs_len=82)
        test_dataloader = utils.DataLoader(test_gen, shuffle=True, batch_size=32, num_workers=8)

    test_loss = 0
    mc_samples = 10
    for batch_idz, test_data in tqdm(enumerate(test_dataloader, 0), total=len(
            test_dataloader)):  # indices,scaled_indices, numax, teff, fe_h, age, tams_age
        test_image = test_data[0].cuda().float()
        test_label = test_data[1].cuda().long().squeeze(1)
        test_dnu = test_data[2].cuda().float()
        test_flag = test_data[3].cuda().float()

        test_image = test_image[test_flag != 1]
        test_label = test_label[test_flag != 1]
        test_dnu = test_dnu[test_flag != 1]

        if len(test_label) < 1:
            continue

        pred_grid = np.empty((mc_samples, test_label.shape[0], 2))
        with torch.no_grad():
            for i in range(mc_samples):
                outputs, kl = trained_model.probforward(test_image, test_dnu)
                pred_grid[i, :] = F.softmax(outputs, dim=1).data.cpu().numpy()
        pred_mean = np.mean(pred_grid, axis=0)
        epistemic = np.mean(pred_grid ** 2, axis=0) - np.mean(pred_grid, axis=0) ** 2
        aleatoric = np.mean(pred_grid * (1 - pred_grid), axis=0)

        pred = np.argmax(pred_mean, axis=1)
        pred_epi = epistemic[np.arange(len(pred)), np.argmax(pred_mean, axis=1)]
        pred_alea = aleatoric[np.arange(len(pred)), np.argmax(pred_mean, axis=1)]

        test_mean.append(pred_mean[np.arange(len(pred)), 1])
        test_pred.append(pred)
        test_aleatoric.append(pred_alea)
        test_epistemic.append(pred_epi)
        test_truth.append(test_label.data.cpu().numpy())
        test_dnu_vec.append(test_dnu)
    test_mean = np.concatenate(test_mean, axis=0)
    test_pred = np.concatenate(test_pred, axis=0)
    test_truth = np.concatenate(test_truth, axis=0)
    test_dnu_vec = np.concatenate(test_dnu_vec, axis=0)
    test_aleatoric = np.concatenate(test_aleatoric, axis=0)
    test_epistemic = np.concatenate(test_epistemic, axis=0)

    print('Accuracy: ', accuracy_score(y_true=test_truth, y_pred=test_pred))
    print('Test Loss: ', test_loss)
    print('Recall: ', recall_score(y_true=test_truth, y_pred=test_pred))
    print('Precision: ', precision_score(y_true=test_truth, y_pred=test_pred))
    print('Confusion Matrix: \n', confusion_matrix(y_true=test_truth, y_pred=test_pred))

    bins = np.linspace(2.0, 9.2, num=37)
    diff_bins = bins[:-1] - 0.5 * np.diff(bins)
    digitized_dnu = np.digitize(test_dnu_vec, bins=bins)
    while len(np.unique(digitized_dnu)) != len(diff_bins):
        diff_bins = np.append(diff_bins, diff_bins[-1] + 0.5 * (diff_bins[1] - diff_bins[0]))
    print('Diff Bins: ', diff_bins)

    accuracy_vector = []
    precision_vector_pos = []
    precision_vector_neg = []
    print('Test Pred Shape: ', test_pred.shape)
    print('Digitized Dnu Shape: ', digitized_dnu.shape)
    test_pred = test_pred.squeeze()
    test_truth = test_truth.squeeze()
    test_mean = test_mean.squeeze()
    digitized_dnu = digitized_dnu.squeeze()
    for val in np.unique(digitized_dnu):
        print('Evaluating Dnu: ', test_dnu_vec[digitized_dnu == val][0])
        select_pred_label = test_pred[digitized_dnu == val]
        select_truth_label = test_truth[digitized_dnu == val]
        select_proba = test_mean[digitized_dnu == val]
        accuracy_vector.append(np.sum(np.abs(select_pred_label - select_truth_label)) / len(select_truth_label))
        precision_vector_pos.append(precision_score(y_pred=select_pred_label, y_true=select_truth_label, pos_label=1))
        precision_vector_neg.append(precision_score(y_pred=select_pred_label, y_true=select_truth_label, pos_label=0))
    accuracy_vector = np.array(accuracy_vector) * 100
    precision_vector_pos = np.array(precision_vector_pos) * 100
    precision_vector_neg = np.array(precision_vector_neg) * 100

    plt.hist(test_mean, bins=50)
    plt.title('Distribution of Predictions')
    plt.xlabel('$p$')
    plt.ylabel('Count')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.bar(diff_bins,
            precision_vector_pos, edgecolor='k', width=np.median(np.diff(bins)))

    ax1.set_ylabel('HeB Precision/Purity (%)')
    ax1.set_xlabel('$\\Delta\\nu$ ($\\mu$Hz)')

    ax2.bar(diff_bins,
            precision_vector_neg, edgecolor='k', width=np.median(np.diff(bins)),
            color='red')

    ax2.set_ylabel('RGB Precision/Purity (%)')
    ax2.set_xlabel('$\\Delta\\nu$ ($\\mu$Hz)')
    ax1.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
    ax2.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18])
    ax1.set_xlim([1.5, 9.7])
    ax2.set_xlim([1.5, 9.7])

    plt.show()
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.scatter(test_mean, np.sqrt(test_epistemic), s=7, c='orange')
    ax1.set_xlabel('Mean Prediction')
    ax1.set_ylabel('Epistemic $\\sigma$')
    ax2.scatter(test_mean, np.sqrt(test_aleatoric), s=7, c='blue')
    print('Length Plotted: ', len(test_aleatoric))
    print('Unique Epistemic: ', len(np.unique(test_epistemic)))
    ax2.set_xlabel('Mean Prediction')
    ax2.set_ylabel('Aleatoric $\\sigma$')

    ax3.scatter(test_mean, np.sqrt(test_epistemic + test_aleatoric), s=7, c='r')
    print('Unique Total: ', len(np.unique(np.sqrt(test_epistemic + test_aleatoric))))

    ax3.set_xlabel('Mean Prediction')
    ax3.set_ylabel('Total $\\sigma$')
    print('Total with sigma above 0.1: ', np.sum(np.sqrt(test_epistemic + test_aleatoric) > 0.1))

    wrong_aleatoric = test_aleatoric[np.abs(test_truth - test_pred) == 1]
    wrong_epistemic = test_epistemic[np.abs(test_truth - test_pred) == 1]
    print(wrong_aleatoric.shape)
    ax4.hist(np.sqrt(wrong_aleatoric), histtype='step', bins=50, color='blue',
             label='Aleatoric $\\sigma$ for Wrong Pred')
    ax4.hist(np.sqrt(wrong_epistemic), histtype='step', bins=50, color='orange',
             label='Epistemic $\\sigma$ for Wrong Pred')
    ax4.hist(np.sqrt(wrong_aleatoric + wrong_epistemic), histtype='step', bins=50, color='red',
             label='Total $\\sigma$ for Wrong Pred')
    ax4.legend()
    ax4.set_ylabel('Count')
    ax4.set_xlabel('$\\sigma$')
    plt.tight_layout(w_pad=3, h_pad=3)
    plt.show()


def train_model_metric_learning():
    model = Simple_Classifier_No_Dnu()
    model.cuda()
    torch.backends.cudnn.benchmark = True
    print('CUDNN Enabled? ', torch.backends.cudnn.enabled)

    folder_kic = []
    root_folder = '/data/marc/AstroBlasto/npz_BGCorr/BGCorrSpectra(4years-Consensus)/'
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filex in filenames:
            folder_kic.append(int(filex.split('-')[0]))

    folder_kic = np.unique(folder_kic)

    #train_kics, test_kics = train_test_split(folder_kic, test_size=0.15, random_state=137)
    #print('Number of Test KICs: ', len(test_kics))

    jie_data = pd.read_csv('/home/z3384751/K2Detection/JieData_Full2018.txt', delimiter='|', header=0)
    jie_kic = jie_data['KICID'].values
    jie_dnu = jie_data['dnu'].values

    label_data = pd.read_csv('/data/marc/AstroBlasto/Elsworth_Jie_2019_ID_Label.dat', header=0, delim_whitespace=True)
    label_kic = label_data['KIC'].values
    label_truth = label_data['Label'].values

    #################### THIS SELECTS THE 'FULL' CATALOG GIVEN BY DENNIS #####################
    vis_cat = '/home/z3384751/AstroBlasto/Stello_Suppression_visibility_Cat.txt'
    vis_df = pd.read_table(vis_cat, header = 0, delim_whitespace=True)
    kic = vis_df['KIC'].values
    numax = vis_df['numax'].values
    int_power_l1 = vis_df.iloc[:,3].values
    int_power_l0 = vis_df.iloc[:,4].values
    dipole_vis = int_power_l1/int_power_l0
    kic = kic[numax >= 50]
    dipole_vis = dipole_vis[numax >= 50]
    numax= numax[numax>= 50]
    numax_array = np.linspace(50, 250, num= 1000)
    fiducial_line = (-0.25/190)*numax_array + (0.75 + 12.5/190)
    below_kic=[]
    below_vis = []
    below_numax = []
    for i in range(len(kic)):
        fiducial_value = (-0.25/190)*numax[i] + (0.75 + 12.5/190)
        if dipole_vis[i] < fiducial_value:
            below_kic.append(kic[i])
            below_numax.append(numax[i])
            below_vis.append(dipole_vis[i])

    below_kic = np.array(below_kic)
    print('Number of suppressed stars: ', len(below_kic))
    for i in range(len(label_kic)):
        if label_kic[i] in below_kic:
            label_truth[i] = 2

    print('Unique Labels: ', np.unique(label_truth))
    print('Total Labels: ', len(label_truth))

    ##########################################################################################

    # x1 = 50, y1 = .75
    # x2 = 240, y2 = .5
    select_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in folder_kic])
    #train_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in train_kics])
    #test_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in test_kics])
    #val_label = np.array([label_truth[np.where(label_kic == kicz)][0] for kicz in val_kics])

    print('Num Label 0: ', np.sum(select_label == 0))
    print('Num Label 1: ', np.sum(select_label == 1))
    print('Num Label 2: ', np.sum(select_label == 2))

    train_kics, val_kics, train_label, val_label = train_test_split(folder_kic,select_label, test_size=0.15, random_state=137, stratify=select_label)
    train_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in train_kics])
    val_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in val_kics])
    #test_dnu = np.array([jie_dnu[np.where(jie_kic == kicz)][0] for kicz in test_kics])

    print('Number of Train KICs: ', len(train_kics))
    print('Number of Validation KICs: ', len(val_kics))
    train_gen = npz_generator(root=root_folder, select_kic=train_kics, obs_len=365*4, external_kic = label_kic, external_label = label_truth) # obs_len in days
    train_dataloader = utils.DataLoader(train_gen, shuffle=True, batch_size=32, num_workers=8)

    val_gen = npz_generator(root=root_folder, select_kic=val_kics, obs_len=365*4, external_kic = label_kic, external_label = label_truth)
    val_dataloader = utils.DataLoader(val_gen, shuffle=True, batch_size=32, num_workers=8)

    train_batch_sampler = BalancedBatchSampler(train_label, n_classes=3, n_samples=10)
    train_dataloader_online = utils.DataLoader(train_gen, num_workers=10, batch_sampler=train_batch_sampler)
    val_batch_sampler = BalancedBatchSampler(val_label, n_classes=3, n_samples=10)
    val_dataloader_online = utils.DataLoader(val_gen, num_workers=10, batch_sampler=val_batch_sampler)

    # plt.hist(test_dnu[test_label == 0], color='r', bins=50, alpha=0.5)
    # plt.hist(test_dnu[test_label == 1], color='b', bins=25, alpha=0.5)
    print('Nb Test Dnu smaller than 9: ', np.sum(val_dnu < 9))
    print('Nb Train Dnu smaller than 9: ', np.sum(train_dnu < 9))
    # plt.yscale('log')
    # plt.show()

    train_loader = train_dataloader_online
    val_loader = val_dataloader_online

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1E-7)
    margin = 1.

    loss_function = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))

    n_epochs = 200
    best_loss = 1.e9
    model_checkpoint = False
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)
        train_loss = 0
        train_batches = 0
        train_kic_array = []

        model.train()  # set to training mode

        for batch_idx, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), unit='batches'):

            image = data[0].cuda().float()
            label = data[1].cuda().long().squeeze(1)
            dnu = data[2].cuda().float()
            flag = data[3].cuda().float()
            train_kic = data[4].cuda().float()

            image = image[flag != 1] # flag 1 is bad stuff, flag 2 is dnu > 9
            label = label[flag != 1]
            dnu = dnu[flag != 1]
            train_kic = train_kic[flag != 1]



            if len(label) < 2:
                print('Insufficient Batch!')
                continue

            model_optimizer.zero_grad()
            outputs = model(image)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            label = (label,)
            loss_inputs += label

            # Calculate loss and backpropagate
            loss_outputs = loss_function(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            train_loss += loss.item()  # Summing losses across all batches, so if you want the mean for EACH sample, divide by number of batches
            train_batches += 1
            train_kic_array.append(train_kic.data.cpu().numpy())
            loss.backward()
            model_optimizer.step()

        train_loss = train_loss / train_batches
        train_kic_array = np.concatenate(train_kic_array, 0)

        print('KICs through Training Pass:', len(np.unique(train_kic_array)))
        print('Total Train KIC in Ambiguous Region: ', np.sum(train_dnu < 9))

        val_loss = 0
        val_batches = 0
        val_kic_array = []
        model.eval()

        with torch.no_grad():
            for batch_idy, val_data in tqdm(enumerate(val_loader, 0), total=len(val_loader), unit='batches'):
                val_image = val_data[0].cuda().float()
                val_label = val_data[1].cuda().long().squeeze(1)
                val_dnu = val_data[2].cuda().float()
                val_flag = val_data[3].cuda().float()
                val_kic = val_data[4].cuda().float()
                val_image = val_image[val_flag != 1] # flag 1 is bad stuff, flag 2 is dnu > 9
                val_label = val_label[val_flag != 1]
                val_dnu = val_dnu[val_flag != 1]
                val_kic = val_kic[val_flag != 1]

                outputs = model(val_image)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)

                loss_inputs = outputs
                val_label = (val_label,)
                loss_inputs += val_label

                # Calculate loss and backpropagate
                loss_outputs = loss_function(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()
                val_batches += 1
            val_loss = val_loss / val_batches
            
        print('\n\nTrain Loss: ', train_loss)
        print('Val Loss: ', val_loss)

        scheduler.step(train_loss)  # reduce LR on loss plateau


        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])

        if model_checkpoint:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best:
                filepath = '/home/z3384751/AstroBlasto/saved_models/OTF_Pytorch/82dConsensusNoVal_LLRGB_ACC:%.2f-Loss:%.3f' % (
                    val_acc, val_loss)
                print('Model saved to %s' % filepath)
                torch.save(model.state_dict(), filepath)
                # to load models:
                # the_model = TheModelClass(*args, **kwargs)
                # the_model.load_state_dict(torch.load(PATH))
            else:
                print('No improvement over the best of %.4f' % best_loss)

        if epoch % 200 == 0:
            train_embeddings_baseline, train_labels_baseline, train_embed_filenames = extract_embeddings(
                train_dataloader, model)
            np.savez_compressed('RBG_RC_Supp_year_Embed2_OnlineTripletTrain_NoDnu_Trial2', embedding=train_embeddings_baseline, label=train_labels_baseline, filename=train_embed_filenames)
            val_embeddings_baseline, val_labels_baseline, val_embed_filenames = extract_embeddings(
                val_dataloader, model)
            np.savez_compressed('RBG_RC_Supp_year_Embed2_OnlineTripletVal_NoDnu_Trial2', embedding=val_embeddings_baseline, label=val_labels_baseline, filename=val_embed_filenames)

            plot_embeddings(train_embeddings_baseline, train_labels_baseline)
            plot_embeddings(val_embeddings_baseline, val_labels_baseline)


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    colors = ['#1f77b4', '#ff7f0e', 'r','#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    classes = ['RGB', 'RC', 'SUPP']

    plt.figure(figsize=(10,10))
    for i in range(3):
        inds = np.where(targets==i)[0]
        print('Inds Length: ', len(inds))
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    plt.show()
    plt.close()

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        flags = np.zeros(len(dataloader.dataset))
        filenames = []
        k = 0
        for images, target, dnu, flag, filename in dataloader:           
            images = images.cuda().float()
            dnu = dnu.cuda().float()
            emb = model.get_embedding(images)
            embeddings[k:k+len(images)] = emb.data.cpu().numpy()
            labels[k:k+len(images)] = np.ravel(target.numpy())
            flags[k:k+len(images)] = flag.numpy()
            k += len(images)
            filenames.append(filename)
        embeddings = embeddings[flags != 1]
        labels = labels[flags != 1]
        filenames = np.concatenate(filenames, axis=0)
        filenames = filenames[flags != 1]
    return embeddings, labels, filenames




train_model()
#train_model_k_fold()
# test_model(test_single=False)
#train_model_metric_learning()
