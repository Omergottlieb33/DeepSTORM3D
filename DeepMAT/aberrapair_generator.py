import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image

from DeepSTORM3D.physics_utils import calc_bfp_grids, EmittersToPhases, PhysicalLayer
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor
from DeepSTORM3D.helper_utils import CalcMeanStd_All

from DeepMAT.utils.cfg_utils import get_image_generator_params
from DeepMAT.utils.aberrations_utils import get_phase_mask_aberration, random_zernike_parameters

ZERNIKE_COEFFICIENTs_MAX_N = 5
ZERNIKE_COEFFICIENTs_MAX_NUM_ABR = 10

class AberrationPairGenerator:
    def __init__(self, image_generator_config, training_config):
        self.params = get_image_generator_params(
            image_generator_config, training_config
        )
        self._initialize()
    
    def _initialize(self):
        """
        Internal initialization method to set up parameters,
        random seeds, and compute grids.
        """
        # Set random seeds for repeatability
        torch.manual_seed(999)
        np.random.seed(566)

        self.device = self.params["device"]
        torch.backends.cudnn.benchmark = True

        # Calculate effective pixel sizes
        self.params["pixel_size_FOV"] = (
            self.params["pixel_size_CCD"] / self.params["M"]
        )  # in [um]
        # in [um]
        self.params["pixel_size_rec"] = self.params["pixel_size_FOV"] / 1

        # Calculate axial range and pixel size
        self.params["axial_range"] = (
            self.params["zmax"] - self.params["zmin"]
        )  # [um]
        self.params["pixel_size_axial"] = (
            self.params["axial_range"] / self.params["D"]
        )  # [um]

        # Precompute back-focal-plane grids
        self.params = calc_bfp_grids(self.params)

        # Create output directory if it doesn't exist
        self.path_train = self.params["training_data_path"]
        if not os.path.isdir(self.path_train):
            os.mkdir(self.path_train)
        self.path_train_gt = os.path.join(self.path_train, "gt/")
        if not os.path.isdir(self.path_train_gt):
            os.mkdir(self.path_train_gt)
        self.path_train_abr = os.path.join(self.path_train, "abr/")
        if not os.path.isdir(self.path_train_abr):
            os.mkdir(self.path_train_abr)

        # Compute number of training/validation batches
        ntrain_batches = int(
            self.params["ntrain"] / self.params["batch_size_gen"])
        nvalid_batches = int(
            self.params["nvalid"] / self.params["batch_size_gen"])
        self.params["ntrain_batches"] = ntrain_batches
        self.params["nvalid_batches"] = nvalid_batches

        # Dictionary to store XYZ labels for all images
        self.labels_dict = {}
        self.mask_abr = {}
    
    def __call__(self):
        mask_init = self.params.get("mask_init", None)
        if mask_init is None:
            raise ValueError("Please supply a phase mask via mask_init.")
        else:
            mask_param = torch.from_numpy(mask_init)

        # Initialize the physical model
        psf_module = PhysicalLayer(self.params)

        # Generate training data
        self._generate_data(
            psf_module=psf_module,
            mask_param=mask_param,
            n_batches=self.params["ntrain_batches"],
            start_idx=0,
            is_train=True,
        )

        self._save_labels()
        self._save_setup_params()
    
    def _generate_data(self, psf_module, mask_param, n_batches, start_idx, is_train):
        """
        Generate either training or validation data.

        Parameters
        ----------
        psf_module : PhysicalLayer
            The physical model used to generate images from emitter phases.
        mask_param : torch.Tensor
            The phase mask parameter.
        n_batches : int
            Number of batches to generate.
        start_idx : int
            Starting index for naming/saving images.
        is_train : bool
            Flag indicating whether this is training data or validation data.
        """
        for i in range(n_batches):
            # Sample emitter positions (xyz) and photons
            xyz, Nphotons = generate_batch(
                self.params["batch_size_gen"], self.params)

            # Convert phases to tensors
            phase_emitters = EmittersToPhases(xyz, self.params)
            phases_tensor = complex_to_tensor(phase_emitters)
            Nphotons_tensor = torch.from_numpy(Nphotons).float()

            # Pass through physical layer
            # calibrate phase mask
            im = psf_module(
                mask_param.to(self.device),
                phases_tensor.to(self.device),
                Nphotons_tensor.to(self.device),
            )
            im_np = np.squeeze(im.data.cpu().numpy())
            # abrrated phase mask
            coefficients, amplitude = random_zernike_parameters(ZERNIKE_COEFFICIENTs_MAX_N, ZERNIKE_COEFFICIENTs_MAX_NUM_ABR)
            mask_abr = get_phase_mask_aberration(mask_param, coefficients, amplitude)
            np.savez_compressed(os.path.join(self.path_train_abr, f'mask_abr{start_idx + i}.npz'), mask_abr)
            mask_param_abr = mask_abr + np.array(mask_param)
            mask_param_abr = torch.from_numpy(mask_param_abr)
            im_abr = psf_module(
                mask_param_abr.to(self.device),
                phases_tensor.to(self.device),
                Nphotons_tensor.to(self.device),
            )
            im_abr_np = np.squeeze(im_abr.data.cpu().numpy())
            # Normalize the image if project_01 is False
            if not self.params["project_01"]:
                mean_val, std_val = self.params["global_factors"]
                im_np = (im_np - mean_val) / std_val
                im_abr_np = (im_abr_np - mean_val) / std_val

            # Threshold dim emitters if counts are gamma distributed and multi-emitter
            if (not self.params["nsig_unif"]) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > self.params["nsig_thresh"], :]

            # Save the image
            img_name = os.path.join(self.path_train_gt, f"im{start_idx + i}.tiff")
            img_obj = Image.fromarray(im_np)
            img_obj.save(img_name)

            img_name_abr = os.path.join(self.path_train_abr, f"im{start_idx + i}.tiff")
            img_obj_abr = Image.fromarray(im_abr_np)
            img_obj_abr.save(img_name_abr)

            # Save corresponding labels
            self.labels_dict[str(start_idx + i)] = xyz

            # Print progress
            if is_train:
                print(
                    f"Training Example [{i+1} / {n_batches}]"
                )
            else:
                print(
                    f"Validation Example [{i+1} / {n_batches}]"
                )

        # Calculate and store training stats only for training data
        if is_train:
            self.params["train_stats"] = CalcMeanStd_All(
                self.path_train_gt, self.labels_dict
            )
    
    def _save_labels(self):
        """Save all XYZ labels as a pickle file."""
        path_labels = os.path.join(self.path_train, "labels.pickle")
        with open(path_labels, "wb") as handle:
            pickle.dump(self.labels_dict, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    
    def _save_setup_params(self):
        """
        Save the setup parameters and partition information
        (train/valid IDs) into a pickle file.
        """
        ntrain = self.params["ntrain_batches"]
        nvalid = self.params["nvalid_batches"]
        total = ntrain + nvalid

        # Create partition info
        indices = list(range(total))
        list_IDs = [str(idx) for idx in indices]
        partition = {
            "train": list_IDs[:ntrain],
            "valid": list_IDs[ntrain:],
        }
        self.params["partition"] = partition

        # Optionally adjust pixel_size_rec if needed
        self.params["pixel_size_rec"] = self.params["pixel_size_FOV"] / 4

        # Save setup params dictionary
        path_setup_params = os.path.join(
            self.path_train, "setup_params.pickle")
        with open(path_setup_params, "wb") as handle:
            pickle.dump(self.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Finished sampling examples!")

if __name__ == "__main__":
    # Example usage
    img_generator = AberrationPairGenerator(
        "DeepMAT/cfg/image_generator_config.yaml",
        "DeepMAT/cfg/train_config.yaml"
    )
    img_generator()