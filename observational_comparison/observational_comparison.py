import my_functions as mf
from my_functions import LoadSimulation
import numpy as np
from scipy.stats import binned_statistic_dd

class ObservationalComparison(LoadSimulation):
    """Adjusted LoadSimulation object, designed to more closely follow observers in the literature.
    Differences:
    - All galaxies in a halo have the same velocity
    - Gaussian smoothing

    Args:
        LoadSimulation (_type_): _description_
    """

    def load_all(self: object, bins: int):
        """Calls load_variables, velocities and bin_galaxies

        Args:
            self (object): simulation object
            bins (int): number of voxels along each axis
        """
        self.load_variables()
        self.bin_galaxies(bins)
        self.velocities()

    def load_variables(self: object):
        """Loads in some basic properties

        Args:
            self (object): simulation object
        """
        self.halo_centers = np.array(self.data.input_halos.halo_centre.to_physical()[self.mask])
        self.vp_true = np.array(self.data.exclusive_sphere_50kpc.centre_of_mass_velocity.to('km/s').to_physical()[self.mask])[:, 0]

        H = self.cosmology_raw['H [internal units]']
        self.halo_centers_z = self.halo_centers.copy()
        self.halo_centers_z[:, 0] = (self.halo_centers_z[:, 0] + self.vp_true / H) % self.boxsize

    def bin_galaxies(self: object, bins: int):
        """Bins galaxies in voxels in 3D and redshift space

        Args:
            self (object): simulation object
            bins (int): number of voxels along each axis
        """
        self.bins = bins
        positions = np.linspace(0, self.boxsize, self.bins+1)
        H = self.cosmology_raw['H [internal units]']
        
        self.number_density, _ = np.histogramdd(self.halo_centers, bins=[positions, positions, positions])
        self.mean_galaxy_number_density = np.mean(self.number_density)
        self.delta_g = self.number_density / self.mean_galaxy_number_density - 1

        velocity_x = (self.halo_centers[:, 0] * H + self.vp_true) % (self.boxsize * H)
        velocity_bins = positions * H
        self.number_density_z = np.histogramdd(np.transpose([velocity_x, *self.halo_centers[:, 1:].T]), [velocity_bins, positions, positions])[0]
        self.delta_g_z = self.number_density_z / self.mean_galaxy_number_density - 1

    def velocities(self: object):
        """Calculates the voxel velocity based on velocities of central galaxies

        Args:
            self (object): simulation object
        """
        self.is_central = self.data.input_halos.is_central[self.mask] == 1

        hostid = np.array(self.data.input_halos_hbtplus.host_fofid[self.mask])
        hostid_central = hostid[self.is_central]
        velocity_central = self.vp_true[self.is_central]

        velocity_adjusted_mapping = np.zeros(np.max(hostid_central)+1)
        velocity_adjusted_mapping[hostid_central] = velocity_central

        velocity_of_host = velocity_adjusted_mapping[hostid]
        self.vp = velocity_of_host

        positions = np.linspace(0, self.boxsize, self.bins+1)
        voxel_count = np.histogramdd(np.array(self.halo_centers), bins=[positions]*3)[0]
        voxel_velocity_sum_of_squares = np.array(np.histogramdd(
            np.array(self.halo_centers),
            bins=[positions]*3, 
            weights=np.array(self.vp)**2
        )[0])

        self.voxel_velocity = np.sqrt(voxel_velocity_sum_of_squares / voxel_count)

        H = self.cosmology_raw['H [internal units]']

        velocity_x = (self.halo_centers[:, 0] * H + self.vp_true) % (self.boxsize * H)
        velocity_bins = positions * H
        voxel_count_z = np.histogramdd(np.transpose([velocity_x, *self.halo_centers[:, 1:].T]), bins=[velocity_bins, positions, positions])[0]
        voxel_velocity_sum_of_squares_z = np.array(np.histogramdd(
            np.transpose([velocity_x, *self.halo_centers[:, 1:].T]),
            bins=[velocity_bins, positions, positions], 
            weights=np.array(self.vp)**2
        )[0])

        self.voxel_velocity_z = np.sqrt(voxel_velocity_sum_of_squares_z / voxel_count_z)

    def delta_g_smoothed(self, r_smooth, redshift_space=True):

        k_i = 2 * np.pi * np.fft.fftfreq(self.bins, self.boxsize / self.bins)
        kx, ky, kz = np.meshgrid(k_i, k_i, k_i)
        k2 = kx**2 + ky**2 + kz**2

        W = np.exp(-.5 * k2 * r_smooth**2)
        
        if redshift_space:
            delta_x = self.delta_g_z
        else:
            delta_x = self.delta_g
        
        delta_k = np.fft.fftn(delta_x)
        delta_k_smoothed = delta_k * W
        delta_x_smoothed = np.real(np.fft.ifftn(delta_k_smoothed))
        return delta_x_smoothed

def save_simulation(simulation_tag, snapshot, mass_tag, n_bins):
    simulation = LoadSimulation(simulation_tag, snapshot, mass_tag)
    simulation.load_all(n_bins)
    delattr(simulation, 'data')
    np.save(f'../storage/simulations_obs/{simulation.simulation}_{simulation.snapshot}_{simulation.mass_tag}_{n_bins}', simulation)

def load_simulation(simulation_tag, snapshot, mass_tag, n_bins):
    return np.load(f'../storage/simulations_obs/{simulation_tag}_{snapshot}_{mass_tag}_{n_bins}.npy', allow_pickle=True).item()