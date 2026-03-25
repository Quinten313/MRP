import swiftsimio as sw
import numpy as np
from dataclasses import dataclass
from scipy.stats import binned_statistic, binned_statistic_dd, skewnorm, t
from scipy.optimize import curve_fit, minimize
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from typing import Callable

class LoadSimulation:
    """This class is used to load in a given simulation. The following methods can be called:
    - select_galaxies_fixed_number OR selext_galaxies_mass_threshold: loads in a selection of galaxies.
    - load_all calls:
        - load_variables: loads in useful variables
        - velocities: calculates central and satellite velocities
        - bin_galaxies: calculates galaxy number density and overdensity
    - calculate_masses: calculates voxel mass and matter overdensity both per voxel and per galaxy

    Args:
        simulation (str): name of simulation
        snapshot (str): index of snapshot
    """
    def __init__(self: object, simulation: str, snapshot: str, mass_tag: str):
        path = f'/net/hydra/data2/quinten/data/{simulation}/SOAP-HBT/halo_properties_{snapshot}.hdf5'
        self.simulation = simulation
        self.snapshot = snapshot
        self.data = sw.load(path)
        self.mass_tag = mass_tag
        self.select_galaxies()
    
    def select_galaxies(self):
        if self.mass_tag[0] == 'N':
            self.select_galaxies_fixed_number()
        else:
            self.selext_galaxies_mass_threshold()

    def select_galaxies_fixed_number(self: object):
        """Loads in the N galaxies with the largest subhalo mass"""

        n = int(self.mass_tag[1:]) * 1_000_000
        halo_mass_all = np.array(self.data.exclusive_sphere_50kpc.total_mass.to('Msun'))
        order = np.argsort(halo_mass_all)
        self.mass_threshold = halo_mass_all[order][-n]
        indices = order[-n:]
        self.mask = np.zeros(len(halo_mass_all), dtype=bool)
        self.mask[indices] = True
        self.hmass = halo_mass_all[self.mask]
    
    def selext_galaxies_mass_threshold(self: object):
        """Loads in all galaxies with subhalo mass within the mass range"""

        self.mass_threshold, self.mass_limit = str_to_mass_range(self.mass_tag)
        halo_mass_all = np.array(self.data.exclusive_sphere_50kpc.total_mass.to('Msun'))
        self.mask = (halo_mass_all >= self.mass_threshold) & (halo_mass_all < self.mass_limit)
        self.hmass = halo_mass_all[self.mask]
        print(f'Mass range: {np.log10(self.mass_threshold)} - {np.log10(self.mass_limit)}\nGalaxies: {np.sum(self.mask)}')

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
        self.boxsize = self.data.metadata.boxsize[0].value
        self.is_central = self.data.input_halos.is_central[self.mask] == 1
        self.halo_centers = np.array(self.data.input_halos.halo_centre.to_physical()[self.mask])
        self.vp = np.array(self.data.exclusive_sphere_50kpc.centre_of_mass_velocity.to('km/s').to_physical()[self.mask])
        self.vp_abs = np.sum(self.vp**2, axis=1)**.5

    def velocities(self: object):
        """Calculates the peculiar velocity of each central and the relative velocity to its host halo for each satellite.
        Also calculates the average velocity of galaxies in each voxel using root mean square.

        Args:
            self (object): simulation object
        """
        hostid = np.array(self.data.input_halos_hbtplus.host_fofid[self.mask])

        hostid_central = hostid[self.is_central]
        velocity_central = self.vp_abs[self.is_central]

        velocity_adjusted_mapping = np.zeros(np.max(hostid_central)+1)
        velocity_adjusted_mapping[hostid_central] = velocity_central

        velocity_of_host = velocity_adjusted_mapping[hostid]
        v_in_halo = self.vp_abs - velocity_of_host
        
        self.v_centrals = velocity_central
        self.v_satellites = v_in_halo[~self.is_central]

        positions = np.linspace(0, self.boxsize, self.bins+1)
        voxel_count = np.histogramdd(np.array(self.halo_centers), bins = [positions]*3)[0]
        voxel_velocity_sum_of_squares = np.array([
            np.histogramdd(np.array(self.halo_centers), bins = [positions]*3, weights=np.array(v)**2)[0] for v in self.vp.T
        ])

        self.voxel_velocity = np.sqrt(voxel_velocity_sum_of_squares / voxel_count)
        self.voxel_velocity_abs = np.sum(self.voxel_velocity**2, axis=0)**.5

        voxel_velocity_squared_std = np.array([
            binned_statistic_dd(
                self.halo_centers, 
                v**2, 
                statistic='std', 
                bins=[positions]*3
            )[0] for v in self.vp.T
        ])
        self.voxel_velocity_err = np.sqrt(voxel_velocity_squared_std / np.sqrt(voxel_count-1))
        self.voxel_velocity_err_abs = np.sum(self.voxel_velocity_err**2, axis=0)**.5

    def bin_galaxies(self: object, bins: int):
        """Calculates the galaxy number density per voxel and adds 
        the galaxy number density and galaxy overdensity per voxel for each galaxy to self

        Args:
            self (object): simulation object
            bins (int): number of voxels along each axis
        """
        self.bins=bins
        positions = np.linspace(0, self.boxsize, self.bins+1)
        self.number_density, edges = np.histogramdd(self.halo_centers, bins=[positions, positions, positions])
        self.galaxy_overdensity = self.number_density / np.mean(self.number_density) - 1
        self.voxel_per_galaxy = np.digitize(self.halo_centers, positions)-1    # Minus 1: np.digitize starts numbering bins at 1
        self.number_density_per_galaxy = self.number_density[*self.voxel_per_galaxy.T]
        self.galaxy_overdensity_per_galaxy = self.number_density_per_galaxy / np.mean(self.number_density) - 1
        self.mean_galaxy_number_density = np.mean(self.number_density)

    def calculate_masses(self: object, path: str):
        """Adds voxel mass and matter overdensity in the form of a three dimensional matrix of size bins x bins x bins.
        The voxel mass and matter overdensity are also output in a one dimensional array of length number of galaxies, 
        containing the voxel mass and matter overdensity of the voxel that the galaxy inhabits.
        The voxel mass should be pre-calculated by calc_voxel_mass based on the snapshot data.

        Args:
            self (object): simulation
            path (str): path to mass density .npy file
        """
        self.voxel_mass = np.load(path)
        self.matter_overdensity_per_voxel = self.voxel_mass / np.mean(self.voxel_mass) - 1
        positions = np.linspace(0, 1000, 101)
        voxel_per_galaxy = np.digitize(self.halo_centers, positions)-1
        self.voxel_mass_per_galaxy = self.voxel_mass[*self.voxel_per_galaxy.T]
        self.matter_overdensity_per_galaxy = (self.voxel_mass_per_galaxy)/np.mean(self.voxel_mass)-1

def save_simulation(simulation_tag, snapshot, mass_tag, n_bins):
    simulation = LoadSimulation(simulation_tag, snapshot, mass_tag)
    simulation.load_all(n_bins)
    delattr(simulation, 'data')
    np.save(f'../storage/simulations/{simulation.simulation}_{simulation.snapshot}_{simulation.mass_tag}_{n_bins}', simulation)

def load_simulation(simulation_tag, snapshot, mass_tag, n_bins):
    return np.load(f'../storage/simulations/{simulation_tag}_{snapshot}_{mass_tag}_{n_bins}.npy', allow_pickle=True).item()

#----------General utility----------
def str_to_mass_range(mass_range):
    mass_range = mass_range.split('_')
    if len(mass_range) == 2:
        return str_to_float(mass_range[0]), str_to_float(mass_range[1])
    elif len(mass_range) == 1:
        if mass_range[0][-1] == '+':
            return str_to_float(mass_range[0][:-1]), np.inf
        elif mass_range[0][-1] == '-':
            return 0, str_to_float(mass_range[0][:-1])
    else:
        raise Exception(f"Invalid notation: {mass_range}")

def str_to_float(x):
    split = x.split('e')
    if len(split) == 1:
        return 10**float(split[0])
    elif len(split) == 2:
        return float(split[0]) * 10**float(split[1])
    else:
        raise Exception(f"Invalid notation: {x}")

def unbias(x: np.ndarray, bias: float):
    """Corrects an overdensity for its bias

    Args:
        x (np.ndarray): array of overdensities + 1
        bias (float): linear bias factor

    Returns:
        x_unbiased (np.ndarray): array of overdensities + 1, corrected for its bias
    """
    return (x-1)/bias+1

def nonlinear_bins(number_density_per_galaxy):
    """Returns an array of bin edges: integer bin edges for
    number density <= 10, log bin edges for number density > 10
    
    Args:
        number_density_per_galaxy (np.ndarray): galaxy number density in corresponding voxel for each galaxy

    Returns:
        bin_edges (np.ndarray): bin edges
    """    
    if np.max(number_density_per_galaxy) <= 10:
        return np.arange(1, np.max(number_density_per_galaxy))
    n_log_bins = int(20*np.log10(np.max(number_density_per_galaxy) / 10))+1
    linear_bins = np.arange(1, 10)
    log_bins = 10**(1+np.arange(n_log_bins+1)/20)
    return np.concatenate([linear_bins, log_bins])

def calc_bin_centers(bin_edges, values):
    """Calculates the mean of a set of values within each bin

    Args:
        bin_edges (np.ndarray): array of bin edges
        values (np.ndarray): values that are binned

    Returns:
        bin_centers (np.ndarray): mean number density within each bin
    """
    bin_numbers = np.digitize(values, bin_edges)-1
    bin_centers = [np.nanmean(values[bin_number == bin_numbers]) for bin_number in range(len(bin_edges)-1)]
    return np.array(bin_centers)

def velocity_function(x: np.ndarray, x0: float, s: float, p: float, c: float) -> np.ndarray:
    """Model of the galaxy velocities as a function of galaxy or matter (number/over)density

    Args:
        x (np.ndarray): galaxy or matter (number/over)density
        x0 (float): fitting parameter related to the transition point between the powerlaws
        s (float): fitting parameter related to the slope of the second powerlaw
        p (float): fitting parameter related to the speed of the transition between the powerlaws
        c (float): fitting parameter related to the velocity plateau that is the first 'powerlaw'

    Returns:
        np.ndarray: velocities predicted by the function
    """
    return (c**p + (x/x0)**(s*p))**(1/p)

def five_point_stencil(y: np.ndarray, boxsize: float, dimension: int, order: int=1):
    """This function approximates the derivative of a property in one dimension using the five point stencil for data with shape (n_bins, n_bins, n_bins).
    
    Args:
        y (np.ndarray): n_bins x n_bins x n_bins sized matrix, containing the values of the property of which the derivative is calculated
        boxsize (float): box size in Mpc
        dimension (int): dimension in which the five point stencil is calculated (e.g. 0 for x, etc.)
        order (int, optional): x'th order derivative. Defaults to first order.
    
    Returns:
        dfdx (np.ndarray): n_bins x n_bins x n_bins sized matrix containing the derivatives
    """
    # Five point stencil formulae
    if order == 1:
        derivative = lambda y1, y2, _, y4, y5: (-y5 + 8*y4 - 8*y2 + y1) / (12*voxel_size)
    elif order == 2:
        derivative = lambda y1, y2, y3, y4, y5: (-y5 + 16*y4 - 30*y3 + 16*y2 - y1) / (12*voxel_size**2)

    # Select correct y_i's given the supplied dimension
    def select_range(range, dimension):
        range3D = [slice(None), slice(None), slice(None)]
        range3D[dimension] = range
        return tuple(range3D)
    
    n_bins = len(y)
    voxel_size = boxsize / n_bins
    indices = np.arange(n_bins)

    range1, range2, range3, range4, range5 = [np.roll(indices, shift) for shift in [2, 1, 0, -1, -2]]
    dfdx = derivative(
        y[select_range(range1, dimension)], 
        y[select_range(range2, dimension)], 
        y[select_range(range3, dimension)], 
        y[select_range(range4, dimension)], 
        y[select_range(range5, dimension)]
    )
    return dfdx


#----------Analysis using mean absolute velocities of individual galaxies----------
def velocity_binned_galaxies(number_density_per_galaxy, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function calculates the mean absolute peculiar velocity, binned in galaxy number density, along with its errors.
    This function also removes any data points with error = 0, as this complicates dealing with errors (e.g. in curve_fit()).

    Args:
        number_density_per_galaxy (np.ndarray): galaxy number density in corresponding voxel for each galaxy
        v (np.ndarray): peculiar velocity of each galaxy

    Returns:
        out (tuple[np.ndarray, np.ndarray, np.ndarray]):
        - **bin_centers** (_np.ndarray_): Mean number density within each bin.
        - **v_mean** (_np.ndarray_): Mean absolute peculiar velocity within each bin.
        - **v_err** (_np.ndarray_): Error on mean absolute peculiar velocity within each bin.
    """
    bin_edges = nonlinear_bins(number_density_per_galaxy)
    bin_centers = calc_bin_centers(bin_edges, number_density_per_galaxy)

    v_mean = binned_statistic(number_density_per_galaxy, np.abs(v), statistic='mean', bins=bin_edges)[0]
    v_std = binned_statistic(number_density_per_galaxy, v, statistic='std', bins=bin_edges)[0]
    v_N = binned_statistic(number_density_per_galaxy, np.abs(v), statistic='count', bins=bin_edges)[0]

    v_mean = np.where(v_std == 0, np.nan, v_mean)
    v_err = np.where(v_std == 0, np.nan, v_std / np.sqrt(v_N))
    return bin_centers, v_mean, v_err

def velocity_binned_matter(matter_overdensity_per_galaxy: np.ndarray, v: np.ndarray, bins: int=30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the mean absolute peculiar velocity, binned 
    in (matter overdensity + 1), along with its errors

    Args:
        matter_overdensity_per_galaxy (np.ndarray): matter overdensity in corresponding voxel
        v (np.ndarray): peculiar velocity of each galaxy
        bins (int, optional): number of matter overdensity bins, defaults to 30

    Returns:
        out (tuple[np.ndarray, np.ndarray, np.ndarray]):
        - **bin_centers** (np.ndarray): mean (matter overdensity + 1) within each bin
        - **v_mean** (np.ndarray): mean absolute peculiar velocity within each bin
        - **v_err** (np.ndarray): error on mean absolute peculiar velocity within each bin
    """
    m = matter_overdensity_per_galaxy + 1
    bin_edges = np.logspace(np.log10(np.min(m)), np.log10(np.max(m)), bins)     # Choose bins s.t. all values fall in a bin
    bin_edges[[0, -1]] = 0, np.inf                                              # Set these limits so the smallest and largest value enter a bin, without risk of rounding errors
    bin_centers = calc_bin_centers(bin_edges, m)

    v_mean = binned_statistic(m, np.abs(v), statistic='mean', bins=bin_edges)[0]
    v_std = binned_statistic(m, v, statistic='std', bins=bin_edges)[0]
    v_N = binned_statistic(m, np.abs(v), statistic='count', bins=bin_edges)[0]
    v_err = v_std / np.sqrt(v_N)
    return bin_centers, v_mean, v_err

def plot_galaxy_overdensity(ax: Axes, number_density_per_galaxy: np.ndarray, mean_galaxy_number_density: float, v: np.ndarray, c: str='black', label: str=None):
    """Plots the mean absolute peculiar velocity in galaxy overdensity bins.

    Args:
        ax (Axes): Axes object to plot on
        number_density_per_galaxy (np.ndarray): galaxy number density in corresponding voxel for each galaxy
        mean_galaxy_number_density (float): mean galaxy number density in the simulation
        v (np.ndarray): peculiar velocities of the galaxies
        c (str, optional): Color of datapoints. Defaults to black
        label (str, optional): Label of datapoints. Defaults to None
    """
    bin_centers, v_mean, v_err = velocity_binned_galaxies(number_density_per_galaxy, v)
    ax.errorbar(bin_centers/mean_galaxy_number_density, v_mean, v_err, linestyle='', c=c, label=label, capsize=1)

def plot_matter_overdensity(ax: Axes, matter_overdensity_per_galaxy: np.ndarray, v: np.ndarray, bins: int=30, c: str='black', label: str=None):
    """Plots the mean absolute peculiar velocity in matter overdensity bins.

    Args:
        ax (Axes): Axes object to plot on
        matter_overdensity_per_galaxy (np.ndarray): matter overdensity in corresponding voxel
        v (np.ndarray): peculiar velocities of the galaxies
        bins (int, optional): Number of matter overdensity bins. Defaults to 30
        c (str, optional): Color of datapoints. Defaults to black
        label (str, optional): Label of datapoints. Defaults to None
    """
    bin_centers, v_mean, v_err = velocity_binned_matter(matter_overdensity_per_galaxy, v, bins=bins)
    ax.errorbar(bin_centers, v_mean, v_err, linestyle='', c=c, label=label)

def fit_velocities_galaxies(number_density_per_galaxy: np.ndarray, mean_galaxy_number_density: float, v: np.ndarray, p0: list=None):
    """Uses curve_fit to find the best fit through the velocities as a function of galaxy overdensity + 1

    Args:
        number_density_per_galaxy (np.ndarray): galaxy number density in corresponding voxel for each galaxy
        mean_galaxy_number_density (int): mean galaxy number density in the simulation
        v (np.ndarray): peculiar velocities of the galaxies
        p0 (list, optional): Initial guess of fitting parameters. Defaults to None
    
    Returns:
        out (tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]):
        - **fit** (np.ndarray): fitting parameters found by scipy.stats.curve_fit
        - **error** (np.ndarray): errors on fitting parameters found by scipy.stats.curve_fit
        - **x_min** (np.float64): smallest value of the galaxy overdensity + 1
        - **x_max** (np.float64): largest value of the galaxy overdensity + 1

    """
    bin_centers, v_mean, v_err = velocity_binned_galaxies(number_density_per_galaxy, v)
    bin_centers = bin_centers / mean_galaxy_number_density                              # Change from number density to density
    mask_nans = ~np.isnan(v_mean)
    fit, covariance = curve_fit(velocity_function, bin_centers[mask_nans], v_mean[mask_nans], sigma=v_err[mask_nans], p0=p0)
    return fit, np.diagonal(covariance)**.5, np.nanmin(bin_centers), np.nanmax(bin_centers)

def plot_fit_galaxies(ax: Axes, number_density_per_galaxy: np.ndarray, mean_galaxy_number_density: float, v: np.ndarray, bias: float=1., p0: list=None, c: str='r', label: str='fit'):
    """Plots a line fitted to the velocities as a function of galaxy overdensity + 1.
    If a bias is given, the galaxy overdensity is corrected for its bias.

    Args:
        ax (Axes): Axes object to plot on
        number_density_per_galaxy (np.ndarray): galaxy number density in corresponding voxel for each galaxy
        mean_galaxy_number_density (int): mean galaxy number density in the simulation
        v (np.ndarray): peculiar velocities of the galaxies
        bias (float, optional): linear bias factor. Defaults to 1
        p0 (list, optional): Initial guess of fitting parameters. Defaults to None
        c (str, optional): Color of fitted line. Defaults to red
        label (str, optional): Label of fitted line. Defaults to fit
    
    Returns:
        out (tuple[np.ndarray,np.ndarray]):
        - **fit** (np.ndarray): fitting parameters found by scipy.stats.curve_fit
        - **error** (np.ndarray): errors on fitting parameters found by scipy.stats.curve_fit
    """
    fit, error, x_min, x_max = fit_velocities_galaxies(number_density_per_galaxy, mean_galaxy_number_density, v, p0=p0)
    xx = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    ax.errorbar(unbias(xx, bias), velocity_function(xx, *fit), c=c, label=label, zorder=-10)
    return fit, error

def plot_fit_matter(ax: Axes, matter_overdensity_per_galaxy: np.ndarray, v: np.ndarray, bins: int=30, p0: list=None, c: str='r', label: str='fit'):
    """Plots a line fitted to the velocities as a function of matter overdensity + 1

    Args:
        ax (Axes): Axes object to plot on
        matter_overdensity_per_galaxy (np.ndarray): matter overdensity in corresponding voxel
        v (np.ndarray): peculiar velocities of the galaxies
        bins (int, optional): Number of matter overdensity bins. Defaults to 30
        p0 (list, optional): Initial guess of fitting parameters. Defaults to None
        c (str, optional): Color of fitted line. Defaults to red
        label (str, optional): Label of fitted line. Defaults to fit
        
    Returns:
        popt (np.ndarray): fitting parameters found by scipy.stats.curve_fit 
    """
    bin_centers, v_mean, v_err = velocity_binned_matter(matter_overdensity_per_galaxy, v)
    mask_nans = ~np.isnan(v_mean)
    fit, covariance = curve_fit(velocity_function, bin_centers[mask_nans], v_mean[mask_nans], sigma=v_err[mask_nans])

    xx = np.logspace(np.log10(np.nanmin(bin_centers)), np.log10(np.nanmax(bin_centers)))
    ax.plot(xx, velocity_function(xx, *fit), c=c, label=label, zorder=-10)
    return fit, np.diagonal(covariance)**.5


#----------Analysis using root mean square voxel velocities----------
@dataclass
class VoxelVelocity:
    """Dataclass containing often used variables

    Args:
        number_density (np.array): n_bins^3 shaped array containing the galaxy number density within each voxel
        n_g_mean (float): mean galaxy number density of all voxels
        voxel_velocity (np.ndarray): n_bins^3 shaped array containing the RMS velocity of each voxel
        voxel_velocity_err (np.ndarray): error on voxel_velocity
        galaxy_overdensity (np.ndarray): n_bins^3 shaped array containing the galaxy overdensity of each voxel
    """
    number_density: np.ndarray
    n_g_mean: float
    voxel_velocity: np.ndarray
    voxel_velocity_err: np.ndarray

    def galaxy_overdensity(self):
        return (self.number_density / self.n_g_mean) - 1

def inter_voxel_errors(delta_g, v, bin_edges, count):
    binned_std_inter = binned_statistic(
        delta_g+1,
        v,
        statistic='std',
        bins=bin_edges,
    )[0]
    binned_err_inter = binned_std_inter / np.sqrt(count-1)
    return binned_err_inter

def intra_voxel_errors(delta_g, v_std, bin_edges, count):
    mask_nans = ~np.isnan(v_std)
    binned_err_intra_sum_of_squares = np.histogram(delta_g[mask_nans]+1, bins=bin_edges, weights=v_std[mask_nans]**2)[0]
    binned_err_intra = np.sqrt(binned_err_intra_sum_of_squares/count) / np.sqrt(count-1)
    binned_err_intra[binned_err_intra == 0] = np.nan
    return binned_err_intra

def binned_voxel_velocity_errors(delta_g, v, v_std, bin_edges, count):
    binned_err_inter = inter_voxel_errors(delta_g, v, bin_edges, count)
    binned_err_intra = intra_voxel_errors(delta_g, v_std, bin_edges, count)
    binned_err = np.sqrt(binned_err_inter**2 + binned_err_intra**2)
    return binned_err, binned_err_inter, binned_err_intra

def bin_voxel_velocity(vv, bin_edges=None):
    """This function bins the RMS voxel velocities in bins (e.g. galaxy overdensity bins or its derivative)
    and calculates the error in each bin based on the inter- and intra-voxel errors in each bin.

    Args:
        vv (VoxelVelocity): dataclass containing information about voxel number density and velocity
        bin_edges (np.array, optional): bin edges to bin the galaxies in delta_g+1 bins. Defaults to None.

    Returns:
        out (tuple[np.ndarray,np.ndarray,np.ndarray,tuple[np.ndarray]]):
        - **bin_centers** (np.ndarray): average x value within a bin
        - **v_binned** (np.ndarray): average RMS velocity in a bin
        - **v_binned_err** (np.ndarray): error on v_binned
        - **error_terms** (tuple[np.ndarray]): inter- and intra-voxel error terms on v_binned
    """
    mask1 = vv.number_density > 0
    mask2 = ~np.isnan(vv.voxel_velocity)
    mask3 = ~np.isnan(vv.voxel_velocity_err)
    mask = mask1&mask2&mask3

    v = vv.voxel_velocity[mask]
    v_err = vv.voxel_velocity_err[mask]
    delta_g = vv.galaxy_overdensity()[mask]

    if type(bin_edges) == type(None):
        bin_edges = nonlinear_bins(vv.number_density) / vv.n_g_mean
    bin_centers = calc_bin_centers(bin_edges, delta_g+1)

    count = np.histogram(delta_g+1, bin_edges)[0]
    v_binned = np.histogram(delta_g+1, bin_edges, weights=v)[0] / count
    v_binned_err, binned_err_inter, binned_err_intra = binned_voxel_velocity_errors(delta_g, v, v_err, bin_edges, count)
    bin_centers[np.isnan(v_binned_err)] = np.nan

    return bin_centers, v_binned, v_binned_err, (binned_err_inter, binned_err_intra)

def plot_voxel_velocity(ax, vv, bin_edges=None, c=None, label=None):
    bin_centers, v_binned, v_binned_err, _ = bin_voxel_velocity(vv, bin_edges)
    ax.errorbar(bin_centers, v_binned, v_binned_err, linestyle='', capsize=1, c=c, label=label)
    return bin_centers

def fit_voxel_velocity(vv, bin_edges=None, p0=None):
    """Fits a model through the voxel velocities.

    Args:
        vv (VoxelVelocity): VoxelVelocity object
        bin_edges (np.ndarray, optional): Bin edges. Defaults to None.
        p0 (np.ndarray, optional): Initial guess. Defaults to None.

    Returns:
        out (tuple):
        - **bin_centers** (np.ndarray): average value within a bin
        - **fit** (np.ndarray): best fitting parameters found by curve_fit()
        - **error** (np.ndarray): errors on fitting parameters
    """
    bin_centers, v_binned, v_binned_err, _ = bin_voxel_velocity(vv, bin_edges)
    mask_nan = (~np.isnan(bin_centers)&~np.isnan(v_binned)&~np.isnan(v_binned_err))
    fit, covariance = curve_fit(velocity_function, bin_centers[mask_nan], v_binned[mask_nan], sigma=v_binned_err[mask_nan], p0=p0)
    return bin_centers, fit, np.diagonal(covariance)**.5

def plot_fit_voxel_velocity(ax, vv, bin_edges=None, p0=None, c=None, label=None, plot_data=None):
    bin_centers, fit, err = fit_voxel_velocity(vv, bin_edges=bin_edges, p0=p0)
    xx = np.logspace(np.log10(np.nanmin(bin_centers)), np.log10(np.nanmax(bin_centers)), 100)
    ax.plot(xx, velocity_function(xx, *fit), c=c, label=label)
    if not plot_data == None:
        try:
            c = plot_data['c']
        except KeyError:
            c = None
        try:
            label = plot_data['label']
        except KeyError:
            label = None
        plot_voxel_velocity(ax, vv, bin_edges=bin_edges, c=c, label=label)
    return fit, err


#----------Skewnormal distributed voxel velocities----------
def exponential(x, a, b, c):
    return a*np.exp(-b*x)+c

def skewnorm7(n_g, params, x):
    return skewnorm.pdf(x, exponential(n_g, *params[:3]), n_g*params[3], exponential(n_g, *params[4:]))

def skewnorm_mean(alpha, xi, omega):
    delta = alpha / np.sqrt(1 + alpha**2)
    return xi + omega*delta*np.sqrt(2/np.pi)

def fit_one_bin_skewnorm(v, p0=[3, 10, 250]):
    mll = lambda args, v=v: -np.sum(np.log(skewnorm.pdf(v, *args)))
    alpha, xi, omega = minimize(mll, p0).x
    return alpha, xi, omega

def model7_to_skewnorm(n_g, args):
    return exponential(n_g, *args[:3]), n_g*args[3], exponential(n_g, *args[4:])

def fit_skewnorm_velocity(n_g, v):
    """Fits a skewnormal distribution as function of numbder density'
    to all data, with alpha as exponential, xi as linear and omega as a constant"""
    v = v[n_g > 1]
    n_g = n_g[n_g > 1]
    mll = lambda args, v=v, n_g=n_g: -np.sum(np.log(skewnorm.pdf(v, exponential(n_g, *args[:-2]), n_g*args[-2], args[-1])))
    minimum = minimize(mll, [30, .1, 3, 5, 250]).x
    return minimum

def fit_seven_parameter_model(n_g, v, p0=None):
    v = v[n_g > 1]
    n_g = n_g[n_g > 1]
    mll = lambda args, v=v, n_g=n_g: -np.sum(np.log(skewnorm.pdf(v, exponential(n_g, *args[:3]), n_g*args[3], exponential(n_g, *args[4:]))))
    if p0 is None:
        p0 = [30, .1, 3, 5, 20, .1, 250]
    minimum = minimize(mll, p0).x
    return minimum

def save_model7(x: LoadSimulation | list[str, str, str, int], p0: list=None):
    if type(x) == list:
        simulation_tag, snapshot, mass_tag, n_bins = x
        simulation = LoadSimulation(simulation_tag, snapshot, mass_tag)
        simulation.load_all(n_bins)
    else:
        simulation = x
        mass_tag, n_bins = simulation.mass_tag, simulation.bins
    model7 = fit_seven_parameter_model(simulation.number_density, simulation.voxel_velocity[0], p0=p0)
    np.save(f'../storage/model7/{simulation.simulation}_{simulation.snapshot}_{simulation.mass_tag}_{n_bins}', model7)

def load_model7(simulation, snapshot, mass_tag, n_bins):
    return np.load(f'../storage/model7/{simulation}_{snapshot}_{mass_tag}_{n_bins}.npy')


#----------Skew-t distributed voxel veloxities----------
def skew_t_pdf(x, alpha, xi, omega, nu):
    return 2/omega * t.pdf((x-xi)/omega, nu) * t.cdf(alpha * (x-xi)/omega * np.sqrt((nu+1) / (nu + ((x-xi)/omega)**2)), nu+1)

def fit_one_bin_skew_t(v, p0=[3, 10, 250, 10]):
    mll = lambda args, v=v: -np.sum(np.log(skew_t_pdf(v, *args)))
    alpha, xi, omega, nu = minimize(mll, p0, bounds=((0, None), (None, None), (0, None), (0, None))).x
    return alpha, xi, omega, nu

def fit_one_bin_skew_t_fixed_param(v, idx, values, p0):
    
    def return_args(args, idx, values):
        arg_list = []
        counter = 0
        for i in range(4):
            if i in idx:
                arg_list.append(values[np.where(np.asarray(idx) == i)[0][0]])
            else:
                arg_list.append(args[counter])
                counter += 1
        return arg_list
        
    mll = lambda args, idx=idx, values=np.array(values): -np.sum(np.log(skew_t_pdf(v, *return_args(args, idx, values))))
    remove_bounds = np.array([True]*4)
    remove_bounds[idx] = False
    bounds = np.array([(0, None), (None, None), (0, None), (0, None)])[remove_bounds]
    minimum = minimize(mll, p0, bounds=bounds).x
    params = np.empty(4)
    counter = 0
    for i in range(4):
        if i in idx:
            params[i] = values[np.where(np.asarray(idx) == i)[0][0]]
        else:
            params[i] = minimum[counter]
            counter += 1
    alpha, xi, omega, nu = params
    return alpha, xi, omega, nu

def fit_all_bins_skew_t(simulation, n_g_min, n_g_max, p0, fix_args=None, print_progress=False):
    parameter_list = []
    n_gs = np.arange(n_g_min, n_g_max)
    for n_g in n_gs:
        if print_progress:
            print(n_g)
        v = simulation.voxel_velocity[0][simulation.number_density == n_g]

        if fix_args:
            indices, functions = fix_args
            for i in range(len(indices)-1):
                if indices[i+1] <= indices[i]:
                    raise ValueError("Argument indices should be asccending")
            parameter_list.append(fit_one_bin_skew_t_fixed_param(v, indices, [f(n_g) for f in functions], p0=p0))
        else:
            parameter_list.append(fit_one_bin_skew_t(v, p0=p0))

        # set p0 for the next bin to the values of the last non-nan non-inf fit
        for fit in parameter_list[::-1]:
            if np.isfinite(np.sum(fit)):
                fit = list(fit)
                if fit[0] > 100:
                    fit[0] = 1
                p0 = fit
                for i in fix_args[0][::-1]:
                    p0.pop(i)
                break
            
    alpha, xi, omega, nu = np.transpose(parameter_list)
    return alpha, xi, omega, nu, n_gs


def plot_model_performance(
        sim: LoadSimulation, 
        pdf: Callable | list[Callable], 
        params: np.ndarray | list[np.ndarray], 
        n_gs: list,
        label: str | list[str] | None = None,
        c: str | list[str] | None  = None,
        path: str | None = None,
    ):
    fig, ax = plt.subplots(1, 3, figsize=[12, 4])
    for i, (axis, n_g) in enumerate(zip(ax, n_gs)):
        v = sim.voxel_velocity[0][sim.number_density == n_g]
        axis.hist(v, bins=100, color='gray', density=True)

        xx = np.linspace(0, np.max(v), 1000)
        if type(pdf) == list:
            for j, (pdf_i, label_i, c_i) in enumerate(zip(pdf, label, c)):
                axis.plot(xx, pdf_i(xx, *params[j][i]), label=label_i, c=c_i)
        else:
            axis.plot(xx, pdf(xx, *params[i]), label=label, c=c)
        axis.set(
            title=f'$n_g$ = {n_g}',
            xlabel='Voxel velocity [km s$^{-1}$]',
        )
    if label:
        ax[0].legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, bbox_inches='tight')
    plt.show()


#----------Calculation and analysis of voxel mass----------
def calc_voxel_mass(path_data: str, output_file: str, bins: int):
    """Calculates the mass density divided in bins^3 cubical voxels using the snapshot data.
    This mass density is written to a .npy file.

    Args:
        path_data (str): path to snapshot data
        output_file (str): name and path of output file
        bins (int): number of voxels along each axis
    """
    boxsize = sw.load(path_data).metadata.boxsize[0]
    step = np.linspace(0, 1, bins+1)*boxsize
    masses3D = []
    
    for i in range(bins):
        region_slice = [
            [step[i], step[i+1]],
            [boxsize*0, boxsize],
            [boxsize*0, boxsize],
        ]
        mask_region = sw.mask(path_data)
        mask_region.constrain_spatial(region_slice)
        snapshot_slice = sw.load(path_data, mask=mask_region)
        
        masses_particles = snapshot_slice.dark_matter.masses
        coords_particles = snapshot_slice.dark_matter.coordinates
        try:
            masses_particles = np.concatenate(
                masses_particles.to('Msun'),
                snapshot_slice.gas.masses.to('Msun'),
                snapshot_slice.stars.masses.to('Msun'), 
                snapshot_slice.dark_matter.masses.to('Msun'),
                )
            coords_particles = np.concatenate(
                coords_particles,
                snapshot_slice.gas.coordinates,
                snapshot_slice.stars.coordinates, 
                snapshot_slice.dark_matter.coordinates,
                )
        except AttributeError:
            pass
        mask = (coords_particles[:, 0] >= step[i]) & (coords_particles[:, 0] < step[i+1])
        masses_particles = masses_particles[mask]
        coords_particles = coords_particles[mask]
        
        pixel_per_particle2D = np.digitize(coords_particles[:, [1,2]], bins=step)-1
        pixel_per_particle = bins*pixel_per_particle2D[:, 0]+pixel_per_particle2D[:, 1]

        mass_grid = np.bincount(pixel_per_particle, masses_particles)
        for _ in range(bins**2-len(mass_grid)):    # Bincount creates an array of the length of the highest voxel index with at least one particle
            mass_grid = np.append(mass_grid, 0)    # The array needs to be padded with zeros to be reshaped if the last voxels are empty
        masses3D.append(mass_grid.reshape([bins, bins]))
        print(i)
    np.save(output_file, np.array(masses3D))

def power_spectrum(simulation: object, overdensity_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the power spectrum of any binned overdensity and removes shot noise.
    This calculation assumes that all particles/galaxies have the same mass.

    Args:
        simulation (object): simulation object
        overdensity_matrix (np.ndarray): three-dimensional matrix containing any overdenisty of a volume

    Returns:
        out (tuple[np.ndarray,np.ndarray]):
        - **k** (np.ndarray): wavenumbers
        - **P(k)** (np.ndarray): corresponding powerspectrum 
    """
    ft = np.fft.fftn(overdensity_matrix)
    ft = np.abs(ft)**2
    N = len(ft[0])
    binsize = round(simulation.boxsize/simulation.bins)
    frequencies = np.fft.fftfreq(N, 1/N)
    k1 = frequencies**2
    k2 = np.array([k1 + freq**2 for freq in frequencies])
    k3 = np.array([k2 + freq**2 for freq in frequencies])
    k = np.round(k3**.5).astype(np.int64)
    P = np.bincount(k.flatten(), weights=ft.flatten())[:N//2] / np.bincount(k.flatten())[:N//2]
    P *= simulation.boxsize**3/simulation.bins**6     # Normalization
    P -= simulation.boxsize**3 / len(simulation.vp_abs)  # Removing shot noise (only works for unweighted particles)
    return 2*np.pi*np.fft.fftfreq(N, binsize)[1:N//2], P[1:]