import my_functions as mf
import swiftsimio
import numpy as np
from scipy.stats import binned_statistic_dd
import os

class ObservationalComparison():
    """Adjusted LoadSimulation object, designed to more closely follow observers in the literature.
    Differences:
    - All galaxies in a halo have the same velocity
    - Gaussian smoothing
    """

    def __init__(self: object, simulation: str, snapshot: str, mass_tag: str):
        path = f'/net/hydra/data2/quinten/data/{simulation}/SOAP-HBT/halo_properties_{snapshot}.hdf5'
        self.simulation = simulation
        self.snapshot = snapshot
        self.data = swiftsimio.load(path)
        self.cosmology = self.data.metadata.cosmology.to_format('mapping')
        self.cosmology_raw = self.data.metadata.cosmology_raw
        self.boxsize = self.data.metadata.boxsize[0].value
        self.mass_tag = mass_tag
        self.select_galaxies()

    def select_galaxies(self: object):
        """Loads in all subhalos within the mass range"""

        if hasattr(self.data.exclusive_sphere_50kpc, 'stellar_mass'):
            print('Subhalo mass: stellar')
            halo_mass_all = np.array(self.data.exclusive_sphere_50kpc.stellar_mass.to('Msun'))
        else:
            print('Subhalo mass: total')
            halo_mass_all = np.array(self.data.exclusive_sphere_50kpc.total_mass.to('Msun'))

        if self.mass_tag[:3] == 'sub':
            self.load_host_halo_mass()
            mass_tag, mass_tag_host = self.mass_tag[3:].split('host')
                        
            mass_threshold_host, mass_limit_host = mf.str_to_mass_range(mass_tag_host)
            mask_fof = (self.halo_mass_200c >= mass_threshold_host) & (self.halo_mass_200c < mass_limit_host)
        else:
            mass_tag = self.mass_tag

        self.mass_threshold, self.mass_limit = mf.str_to_mass_range(mass_tag)
        self.mask = (halo_mass_all >= self.mass_threshold) & (halo_mass_all < self.mass_limit)
        if self.mass_threshold == 1:
            self.mask = self.mask | (halo_mass_all == 0)

        if self.mass_tag[:3] == 'sub':
            if mass_threshold_host > 1:
                self.mask = self.mask & mask_fof
            self.hmass_fof = self.halo_mass_200c[self.mask]
            print(f'Host halo mass range: {np.log10(mass_threshold_host)} - {np.log10(mass_limit_host)}')

        self.hmass = halo_mass_all[self.mask]
        print(f'Subhalo mass range: {np.log10(self.mass_threshold)} - {np.log10(self.mass_limit)}\nGalaxies: {np.sum(self.mask)}')

    def load_host_halo_mass(self):
        host_halo_mass_raw = self.data.spherical_overdensity_200_crit.total_mass.to('Msun')
        hostid = np.array(self.data.input_halos_hbtplus.host_fofid)
        is_central = self.data.input_halos.is_central

        host_halo_mass_adjusted_mapping = np.zeros(np.max(hostid+2))
        host_halo_mass_adjusted_mapping[hostid[is_central == 1]] = host_halo_mass_raw[is_central == 1]
        self.halo_mass_200c = host_halo_mass_adjusted_mapping[hostid]

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

        self.voxel_per_galaxy = np.digitize(self.halo_centers_z, np.linspace(0, self.boxsize, self.bins+1))-1
        self.voxel_per_galaxy_z = np.digitize(self.halo_centers_z, np.linspace(0, self.boxsize, self.bins+1))-1

    def velocities(self: object):
        """Calculates the voxel velocity based on velocities of central galaxies

        Args:
            self (object): simulation object
        """
        self.is_central = self.data.input_halos.is_central[self.mask] == 1

        hostid = np.array(self.data.input_halos_hbtplus.host_fofid[self.mask])
        hostid_central = hostid[self.is_central]
        velocity_central = self.vp_true[self.is_central]

        velocity_adjusted_mapping = np.zeros(np.max(hostid_central)+2)  # +1 such that the maximum value fits: if max is 100, we need np.zeros(101)
        velocity_adjusted_mapping[hostid_central] = velocity_central    # +2 such that haloless centrals (hostid == -1) all get mapped to an unused slot

        self.vp = self.vp_true.copy()
        self.vp[~self.is_central] = velocity_adjusted_mapping[hostid[~self.is_central]]

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
    simulation = ObservationalComparison(simulation_tag, snapshot, mass_tag)
    simulation.load_all(n_bins)
    delattr(simulation, 'data')
    np.save(f'../storage/simulations_obs/{simulation.simulation}_{simulation.snapshot}_{simulation.mass_tag}_{n_bins}', simulation)

def load_simulation(simulation_tag, snapshot, mass_tag, n_bins, allow_save=True):
    path = f'../storage/simulations_obs/{simulation_tag}_{snapshot}_{mass_tag}_{n_bins}.npy'
    
    if os.path.exists(path):
        print('Loading simulation...')
        return np.load(path, allow_pickle=True).item()
    
    elif allow_save:
        print('Saving simulation...')
        save_simulation(simulation_tag, snapshot, mass_tag, n_bins)
        print('Loading simulation...')
        return np.load(path, allow_pickle=True).item()
    
    else:
        print('File not found')

def reconstruct_velocities(simulation, r_smooth, redshift_space=True):
    "Solves the LCE according to the fiducial FLAMINGO cosmology"
    
    if redshift_space:
        delta_g = simulation.delta_g_z
    else:
        delta_g = simulation.delta_g

    f, aH = 0.304611**.55, 68.1
    k_i = 2 * np.pi * np.fft.fftfreq(simulation.bins, simulation.boxsize / simulation.bins)
    kx = k_i[None, :, None]
    ky = k_i[:, None, None]
    kz = k_i[None, None, :]
    k2 = kx**2 + ky**2 + kz**2

    delta_k = np.fft.fftn(delta_g)
    W = np.exp(-.5 * k2 * r_smooth**2)
    v_k = f * aH * delta_k * 1j * ky / k2 * W
    v_k[k2 == 0] = 0

    reconstructed_cube = np.real(np.fft.ifftn(v_k))
    return reconstructed_cube

def get_voxel_velocities_1D(simulation, r_smooth):
    reconstructed_cube = reconstruct_velocities(simulation, r_smooth)

    v_rec_voxel = np.abs(reconstructed_cube)
    v_true_voxel = simulation.voxel_velocity_z

    return v_rec_voxel, v_true_voxel

def get_galaxy_velocities(simulation, r_smooth, redshift_space=True, interpolation=True):
    reconstructed_cube = reconstruct_velocities(simulation, r_smooth, redshift_space)

    if redshift_space:
        halo_centers = simulation.halo_centers_z
        voxel_per_galaxy = simulation.voxel_per_galaxy_z
    else:
        halo_centers = simulation.halo_centers
        voxel_per_galaxy = simulation.voxel_per_galaxy
    
    if interpolation:
        v_rec_galaxy = trilinear_interpolation(reconstructed_cube, halo_centers, simulation.bins, simulation.boxsize/simulation.bins)
    else:
        v_rec_galaxy = reconstructed_cube[*voxel_per_galaxy.T]
    
    v_true_galaxy = simulation.vp

    return v_rec_galaxy, v_true_galaxy

def trilinear_interpolation(voxel_values, coords, n_voxels, voxel_size):
    normalized_positions = ((coords - 0.5 * voxel_size) / voxel_size) % n_voxels
    voxel_idx = np.astype(normalized_positions, int)
    relative_position = normalized_positions - voxel_idx

    x0 = voxel_idx[:, 0]
    y0 = voxel_idx[:, 1]
    z0 = voxel_idx[:, 2]

    x1 = (x0 + 1) % n_voxels
    y1 = (y0 + 1) % n_voxels
    z1 = (z0 + 1) % n_voxels

    x = relative_position[:, 0]
    y = relative_position[:, 1]
    z = relative_position[:, 2]

    interpolated_values = (1-z) * ((1-y) * ((1-x) * voxel_values[x0, y0, z0] + x * voxel_values[x1, y0, z0])\
                        + y * ((1-x) * voxel_values[x0, y1, z0] + x * voxel_values[x1, y1, z0]))\
                        + z * ((1-y) * ((1-x) * voxel_values[x0, y0, z1] + x * voxel_values[x1, y0, z1])\
                        + y * ((1-x) * voxel_values[x0, y1, z1] + x * voxel_values[x1, y1, z1]))

    return interpolated_values

def calc_correlation(array1, array2):
    covariance_matrix = np.cov(array1, array2)
    correlation = covariance_matrix[0, 1] / np.prod(np.diagonal(covariance_matrix))**.5
    return correlation

def calc_correlation_per_density(array1, array2, n_g):
    n_g_max = int(np.max(n_g))
    correlation = np.zeros(n_g_max)
    for n in range(1, n_g_max+1):
        correlation[n-1] = calc_correlation(array1[n_g == n], array2[n_g == n])
    return np.arange(1, n_g_max+1), correlation