import scipy.io as sio
import yaml
import argparse
from PIL import Image
import pickle
import os
from DeepSTORM3D.helper_utils import CalcMeanStd_All
from DeepSTORM3D.physics_utils import calc_bfp_grids, EmittersToPhases, PhysicalLayer
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib
from math import pi
matplotlib.use("TkAgg")


def load_image_generator_parameters(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)

    mask_dict = sio.loadmat(params['mask_path'])
    mask_name = list(mask_dict.keys())[3]
    mask_init = mask_dict[mask_name]
    mask_opts = {'mask_init': mask_init, 'learn_mask': params['learn_mask']}

    optics_dict = {'lamda': params['lamda'],
                   'NA': params['NA'],
                   'noil': params['noil'],
                   'nwater': params['nwater'],
                   'pixel_size_CCD': params['pixel_size_CCD'],
                   'pixel_size_SLM': params['pixel_size_SLM'],
                   'M': params['M'],
                   'f_4f': params['f_4f']}

    data_dims_dict = {'Hmask': params['Hmask'],
                      'Wmask': params['Wmask'],
                      'H': params['H'],
                      'W': params['W'],
                      'clear_dist': params['clear_dist'],
                      'zmin': params['zmin'],
                      'zmax': params['zmax'],
                      'NFP': params['NFP'],
                      'D': params['D']}

    num_particles_dict = {'num_particles_range': params['num_particles_range']}

    nsig_dict = {'nsig_unif': params['nsig_unif'],
                 'nsig_unif_range': params['nsig_unif_range'],
                 'nsig_gamma_params': params['nsig_gamma_params'],
                 'nsig_thresh': params['nsig_thresh']}

    blur_dict = {'blur_std_range': params['blur_std_range']}
    if isinstance(params['nonunif_bg_theta_range'][0], str) and 'pi' in params['nonunif_bg_theta_range'][0]:
        params['nonunif_bg_theta_range'] = [eval(theta.replace('pi', str(pi))) for theta in params['nonunif_bg_theta_range']]
    nonunif_bg_dict = {'nonunif_bg_flag': params['nonunif_bg_flag'],
                       'unif_bg': params['unif_bg'],
                       'nonunif_bg_offset': params['nonunif_bg_offset'],
                       'nonunif_bg_minvals': params['nonunif_bg_minvals'],
                       'nonunif_bg_theta_range': params['nonunif_bg_theta_range']}

    read_noise_dict = {'read_noise_flag': params['read_noise_flag'],
                       'read_noise_nonuinf': params['read_noise_nonuinf'],
                       'read_noise_baseline_range': params['read_noise_baseline_range'],
                       'read_noise_std_range': params['read_noise_std_range']}

    norm_dict = {'project_01': params['project_01'],
                 'global_factors': params['global_factors']}
    batch_gen_dict = {'batch_size_gen': params['batch_size_gen']}

    return {**mask_opts, **num_particles_dict, **nsig_dict, **blur_dict, **nonunif_bg_dict, **read_noise_dict,
            **norm_dict, **optics_dict, **data_dims_dict, **batch_gen_dict}


def load_training_parameters(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f)

    training_dict = {'ntrain': params['ntrain'],
                     'nvalid': params['nvalid'],
                     'training_data_path': params['training_data_path'],
                     'visualize': params['visualize']}

    learning_dict = {'results_path': params['results_path'],
                     'dilation_flag': params['dilation_flag'],
                     'batch_size': params['batch_size'],
                     'max_epochs': params['max_epochs'],
                     'initial_learning_rate': params['initial_learning_rate'],
                     'scaling_factor': params['scaling_factor']}

    checkpoint_dict = {'resume_training': params['resume_training'],
                       'num_epochs_resume': params['num_epochs_resume'],
                       'checkpoint_path': params['checkpoint_path']}

    device = torch.device(
        "cuda:" + str(params['device_id']) if torch.cuda.is_available() else "cpu")

    # device dictionary
    device_dict = {'device': device}

    return {**training_dict, **learning_dict, **checkpoint_dict, **device_dict}


def get_image_generator_params(image_generator_config, training_config):
    image_generator_params = load_image_generator_parameters(
        image_generator_config)
    training_params = load_training_parameters(training_config)
    return {**image_generator_params, **training_params}


class ImageGenerator:
    def __init__(self, image_generator_config, training_config):
        self.params = get_image_generator_params(
            image_generator_config, training_config)
        self.init()

    def init(self):
        # random seed for repeatability
        torch.manual_seed(999)
        np.random.seed(566)
        self.device = self.params['device']
        torch.backends.cudnn.benchmark = True
        # calculate the effective sensor pixel size taking into account magnification and set the recovery pixel size to be
        # the same such that sampling of training positions is performed on this coarse grid
        self.params['pixel_size_FOV'] = self.params['pixel_size_CCD'] / \
            self.params['M']  # in [um]
        # in [um]
        self.params['pixel_size_rec'] = self.params['pixel_size_FOV'] / 1

        # calculate the axial range and the axial pixel size depending on the volume discretization
        self.params['axial_range'] = self.params['zmax'] - \
            self.params['zmin']  # [um]
        self.params['pixel_size_axial'] = self.params['axial_range'] / \
            self.params['D']  # [um]
        # calculate back focal plane grids and needed terms for on the fly PSF calculation
        self.params = calc_bfp_grids(self.params)

        self.path_train = self.params['training_data_path']
        if not (os.path.isdir(self.path_train)):
            os.mkdir(self.path_train)

        # calculate the number of training batches to sample
        ntrain_batches = int(
            self.params['ntrain'] / self.params['batch_size_gen'])
        self.params['ntrain_batches'] = ntrain_batches

        nvalid_batches = int(
            self.params['nvalid'] // self.params['batch_size_gen'])
        self.params['nvalid_batches'] = nvalid_batches

    def __call__(self):
        # generate training data
        mask_init = self.params['mask_init']
        if mask_init is None:
            raise ValueError('supply a phase mask')
        else:
            mask_param = torch.from_numpy(mask_init)
            psf_module = PhysicalLayer(self.params)
        self.labels_dict = {}
        # generate examples for training
        self.generate_train_data(psf_module, mask_param)
        self.generate_valid_data(psf_module, mask_param)
        self.save_labels()
        self.save_setup_params()

    def generate_train_data(self, psf_module, mask_param):
        for i in range(self.params['ntrain_batches']):
            # sample a training example
            xyz, Nphotons = generate_batch(
                self.params['batch_size_gen'], self.params)
            # calculate phases from simulated locations
            phase_emitters = EmittersToPhases(xyz, self.params)
            # cast phases and number of photons to tensors
            Nphotons_tensor = torch.from_numpy(
                Nphotons).type(torch.FloatTensor)
            phases_tensor = complex_to_tensor(phase_emitters)
            # pass them through the physical layer to get the corresponding image
            im = psf_module(mask_param.to(self.device), phases_tensor.to(
                self.device), Nphotons_tensor.to(self.device))
            im_np = np.squeeze(im.data.cpu().numpy())
            # normalize image according to the global factors assuming it was not projected to [0,1]
            if self.params['project_01'] is False:
                im_np = (im_np - self.params['global_factors']
                         [0]) / self.params['global_factors'][1]
            # look at the image if specified
            if self.params['visualize']:
                self.visualize_generated_image(xyz, im_np, i)
             # threshold out dim emitters if counts are gamma distributed
            if (self.params['nsig_unif'] is False) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > self.params['nsig_thresh'], :]
            # save image as a tiff file and xyz to labels dict
            im_name_tiff = self.path_train + 'im' + str(i) + '.tiff'
            img1 = Image.fromarray(im_np)
            img1.save(im_name_tiff)
            self.labels_dict[str(i)] = xyz
        # print number of example
        print('Training Example [%d / %d]' %
              (i + 1, self.params['ntrain_batches']))
        self.params['train_stats'] = CalcMeanStd_All(
            self.path_train, self.labels_dict)

    def generate_valid_data(self, psf_module, mask_param):
        # set the number of particles to the middle of the range for validation
        self.num_particles_range_setup = self.params['num_particles_range']
        self.params['num_particles_range'] = [
            self.num_particles_range_setup[1]//2, self.num_particles_range_setup[1]//2 + 1]
        # sample validation examples
        for i in range(self.params['nvalid_batches']):
            # sample a training example
            xyz, Nphotons = generate_batch(
                self.params['batch_size_gen'], self.params)
            # calculate phases from simulated locations
            phase_emitters = EmittersToPhases(xyz, self.params)
            # cast phases and number of photons to tensors
            Nphotons_tensor = torch.from_numpy(
                Nphotons).type(torch.FloatTensor)
            phases_tensor = complex_to_tensor(phase_emitters)
            # pass them through the physical layer to get the corresponding image
            im = psf_module(mask_param.to(self.device), phases_tensor.to(
                self.device), Nphotons_tensor.to(self.device))
            im_np = np.squeeze(im.data.cpu().numpy())
            # normalize image according to the global factors assuming it was not projected to [0,1]
            if self.params['project_01'] is False:
                im_np = (im_np - self.params['global_factors']
                         [0]) / self.params['global_factors'][1]
            if self.params['visualize']:
                self.visualize_generated_image(xyz, im_np, i)
            if (self.params['nsig_unif'] is False) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > self.params['nsig_thresh'], :]
            # save image as a tiff file and xyz to labels dict
            im_name_tiff = self.path_train + 'im' + \
                str(i + self.params['ntrain_batches']) + '.tiff'
            img1 = Image.fromarray(im_np)
            img1.save(im_name_tiff)
            self.labels_dict[str(i + self.params['ntrain_batches'])] = xyz
         # print number of example
        print('Validation Example [%d / %d]' %
              (i + 1, self.params['ntrain_batches']))

    def save_labels(self):
        # save all xyz's dictionary as a pickle file
        path_labels = self.path_train + 'labels.pickle'
        with open(path_labels, 'wb') as handle:
            pickle.dump(self.labels_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def save_setup_params(self):
        # set the number of particles back to the specified range
        self.params['num_particles_range'] = self.num_particles_range_setup

        # partition built in simulation
        ind_all = np.arange(
            0, self.params['ntrain_batches'] + self.params['nvalid_batches'], 1)
        list_all = ind_all.tolist()
        list_IDs = [str(i) for i in list_all]
        train_IDs = list_IDs[:self.params['ntrain_batches']]
        valid_IDs = list_IDs[self.params['ntrain_batches']:]
        partition = {'train': train_IDs, 'valid': valid_IDs}
        self.params['partition'] = partition

        # in [um]
        self.params['pixel_size_rec'] = self.params['pixel_size_FOV'] / 4

        # save setup parameters dictionary for training and testing
        path_setup_params = self.path_train + 'setup_params.pickle'
        with open(path_setup_params, 'wb') as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Finished sampling examples!')
        # close figure if it was open for visualization
        if self.params['visualize']:
            plt.close(self.fig1)

    def visualize_generated_image(self, xyz, im_np, i):
        # squeeze batch dimension in xyz
        xyz2 = np.squeeze(xyz, 0)
        # plot the image and the simulated xy centers on top
        self.fig1 = plt.figure(1)
        imfig = plt.imshow(im_np, cmap='gray')
        pixel_size_FOV, W, H = self.params['pixel_size_FOV'], self.params['W'], self.params['H']
        plt.plot(xyz2[:, 0] / pixel_size_FOV + np.floor(W / 2),
                 xyz2[:, 1] / pixel_size_FOV + np.floor(H / 2), 'r+')
        plt.title(str(i))
        self.fig1.colorbar(imfig)
        plt.draw()
        plt.pause(0.05)
        plt.clf()


if __name__ == '__main__':
    img_generator = ImageGenerator(
        'DeepMAT/image_generator_config.yaml', 'DeepMAT/train_config.yaml')
    img_generator()
    print('done')
