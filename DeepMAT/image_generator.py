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

# Use "TkAgg" only if you need interactive plots.
matplotlib.use("TkAgg")


class ImageGenerator:
    """
    A class to generate synthetic training and validation image datasets
    by simulating emitters and passing them through a physical model.
    """

    def __init__(self, image_generator_config, training_config):
        """
        Initialize the ImageGenerator.

        Parameters
        ----------
        image_generator_config : str
            Path to the image generator configuration file.
        training_config : str
            Path to the training configuration file.
        """
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

        # Compute number of training/validation batches
        ntrain_batches = int(
            self.params["ntrain"] / self.params["batch_size_gen"])
        nvalid_batches = int(
            self.params["nvalid"] / self.params["batch_size_gen"])
        self.params["ntrain_batches"] = ntrain_batches
        self.params["nvalid_batches"] = nvalid_batches

        # Dictionary to store XYZ labels for all images
        self.labels_dict = {}

    def __call__(self):
        """
        Main entry point for data generation.
        Generates training data, validation data,
        then saves labels and setup parameters.
        """
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

        # Generate validation data
        # Temporarily adjust num_particles_range for validation
        original_num_particles_range = self.params["num_particles_range"]
        midpoint = original_num_particles_range[1] // 2
        self.params["num_particles_range"] = [midpoint, midpoint + 1]
        self._generate_data(
            psf_module=psf_module,
            mask_param=mask_param,
            n_batches=self.params["nvalid_batches"],
            start_idx=self.params["ntrain_batches"],
            is_train=False,
        )
        # Restore original num_particles_range
        self.params["num_particles_range"] = original_num_particles_range

        # Finalize and save meta-information
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
            im = psf_module(
                mask_param.to(self.device),
                phases_tensor.to(self.device),
                Nphotons_tensor.to(self.device),
            )
            im_np = np.squeeze(im.data.cpu().numpy())

            # Normalize the image if project_01 is False
            if not self.params["project_01"]:
                mean_val, std_val = self.params["global_factors"]
                im_np = (im_np - mean_val) / std_val

            # Optionally visualize
            if self.params["visualize"]:
                self._visualize_generated_image(xyz, im_np, i)

            # Threshold dim emitters if counts are gamma distributed and multi-emitter
            if (not self.params["nsig_unif"]) and (xyz.shape[1] > 1):
                Nphotons = np.squeeze(Nphotons)
                xyz = xyz[:, Nphotons > self.params["nsig_thresh"], :]

            # Save the image
            img_name = os.path.join(self.path_train, f"im{start_idx + i}.tiff")
            img_obj = Image.fromarray(im_np)
            img_obj.save(img_name)

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
                self.path_train, self.labels_dict
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

        # Close figure if it was open for visualization
        if self.params["visualize"]:
            plt.close(self.fig1)

    def _visualize_generated_image(self, xyz, im_np, i):
        """
        Visualize the generated image with emitter positions overlaid.
        """
        # Ensure correct shape
        xyz2 = np.squeeze(xyz, 0)  # remove batch dimension
        W, H = self.params["W"], self.params["H"]
        pixel_size_FOV = self.params["pixel_size_FOV"]

        if not hasattr(self, "fig1"):
            self.fig1 = plt.figure()

        plt.figure(self.fig1.number)
        plt.clf()

        imfig = plt.imshow(im_np, cmap="gray")
        plt.title(f"Sample {i}")
        plt.colorbar(imfig)

        plt.plot(
            xyz2[:, 0] / pixel_size_FOV + np.floor(W / 2),
            xyz2[:, 1] / pixel_size_FOV + np.floor(H / 2),
            "r+",
        )
        plt.draw()
        plt.pause(0.05)


if __name__ == "__main__":
    # Example usage
    img_generator = ImageGenerator(
        "DeepMAT/cfg/image_generator_config.yaml",
        "DeepMAT/cfg/train_config.yaml"
    )
    img_generator()
