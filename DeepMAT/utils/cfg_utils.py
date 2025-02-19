import yaml
import scipy.io as sio
import torch
from math import pi



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