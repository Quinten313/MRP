import swiftsimio as sw
import numpy as np

class LoadSimulation:
    def __init__(self, path, mass_threshold, bins):
        self.mass_threshold = mass_threshold
        self.bins = bins
        self.data = sw.load(path)
        self.LoadVariables()
        self.Velocities()
        self.BinGalaxies()
    
    def LoadVariables(self):
        halo_mass_all = self.data.exclusive_sphere_50kpc.total_mass.to('Msun')
        self.mask = halo_mass_all > self.mass_threshold
        self.boxsize = self.data.metadata.boxsize[0].value
        self.is_central = self.data.input_halos.is_central[self.mask] == 1
        self.hmass = halo_mass_all[self.mask]
        self.halo_centers = self.data.input_halos.halo_centre.to_physical()[self.mask]
        vp = self.data.exclusive_sphere_50kpc.centre_of_mass_velocity.to('km/s').to_physical()[self.mask]
        self.vpx, self.vpy, self.vpz = vp[:, 0], vp[:, 1], vp[:, 2]

    def Velocities(self):
        hostid = self.data.input_halos_hbtplus.host_fofid[self.mask]

        hostid_central = hostid[self.is_central]
        velocity_central = self.vpx[self.is_central]

        velocity_adjusted_mapping = np.zeros(np.max(hostid_central)+1)
        velocity_adjusted_mapping[hostid_central] = velocity_central

        velocity_of_host = velocity_adjusted_mapping[hostid]
        v_in_halo = self.vpx.value - velocity_of_host
        
        self.v_centrals = velocity_central               # Peculiar velocity of each central halo
        self.v_satellites = v_in_halo[~self.is_central]  # Difference in peculiar velocity between each satellite and its host

    def BinGalaxies(self):
        positions = np.linspace(0, self.boxsize, self.bins+1)
        H, edges = np.histogramdd(self.halo_centers.value, bins=[positions, positions, positions])
        voxel_per_galaxy = np.digitize(self.halo_centers, positions)-1    # Minus 1: np.digitize starts numbering bins at 1
        self.galaxy_density = np.array([H[*voxel_per_galaxy[i]] for i in range(len(voxel_per_galaxy))])