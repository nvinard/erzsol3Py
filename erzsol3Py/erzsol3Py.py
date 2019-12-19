"""

This script contains functions to create the necessary inputs used by
the software erzsol3. It also contains functions to convert erzsol3s binary
outputs to and hdf5 database.

Contains the following functions:

model2Erzsol3mod:           Generate erzsol model
write2dst:                  Write .dst file (range & azimuth)
write_dst2cmd:              Write dst path to cmd file
MT_components:              Compute full MT from strike, dip, rake and seismic moment
write_mod2cmd:              Write the erzsol3 model path to the cmd file
writeMT2cmd:                Write the moment tensor to the cmd file
writeErzsolOut2cmd:         Write erzsol3s output file name to cmd file
write_betaInfo2cmd:         Write some beta info to cmd file
cart2polar:                 compute range and azimuth from cartesian coordinates
Erzsol3Tohdf5:              Write erzsol output to hdf5 database
readingHdf5:                Read hdf4 file and return data as np.array
points_in_cube:             Compute random locations within some volume
plotClusters:               3-d scatter plot to visualize clusters
randomRGB:                  Compute random RBG values for colors
Erzsol3ToMultipleHdf5:      write single hdf5 outputs of the erzol outputs

Written by Nicolas Vinard, 2019

v0.1

Todo: Extend to write tfrecords

"""

import numpy as np
import h5py
import math
import pandas
import random
from os import listdir
from os.path import isfile, join
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#########################################################
# FUNCTION USED TO CREATE ERZSOL3 INPUTS
#########################################################

def model2Erzsol3mod(csv_file_model: str, erzsol_mod_file: str):

    '''
    model2Erzsol3mod(csv_file_model, erzsol_mod_file)

    This function creates a .mod file which is used for erzsol3.
    Currently it takes only P-velocities and sets the S-velocity to 1 m/s
    in all layers. It is also currently stricly defined for the texas model
    that I used for my CNN application. However, it can easily be changed
    for other purposes and to include S-velocities.

    Parameters
    ----------
    csv_file_model: str
        The path to the csv file containing the P-velocities and depth information
    erzsol_mod_file: str
        The name of the output file ending with .mod

    Returns
    -------
    Writes the erzsol_mod_file

    '''

    model = pandas.read_csv(csv_file_model)
    velocity = model['Velocity'].values
    depth_model = np.arange(0,3450,50)

    # Change velocity to be in 50 m layers instead of 25m
    vel=[]
    for i,k in enumerate(velocity):
        if i%2 == 0:
            vel.append(np.mean(velocity[i:i+1]))

    velocity = np.round(vel)/1000. # velocity in km/s

    # Define constants in dst file:
    dz = (depth_model[1]-depth_model[0])/1000 # convert to km
    vs = 1.0/1000.               # Set shear velocity close to zero (unknown, only interested in P-waves)
    rho = 2.7              # Denisty is unknown, set to a constant
    qa = 0.0               # Attenuation unknown, set to 0
    qb = 0.0               # Attenuation unknown, set to 0

    file = open(erzsol_mod_file, 'w+')

    model_name = 'texasModel\n'
    nLayers_thickness = '    {}        {}\n'.format(len(depth_model), 0)
    L = [model_name, nLayers_thickness]
    file.writelines(L)

    for i, vp in enumerate(velocity):
        L = ['3   {:.3f}    {:.3f}     {:.2f}     {:.3f}    {:.3f}     {:.3f}\n'.format(vp,vs,rho,dz,qa,qb)]
        file.writelines(L)

    file.close()


def write2dst(receiver_file: str, model_file: str, source_coord: np.ndarray, dst_file: str):

    '''
    write2dst(receiver_file, model_file, source_coord, dst_file)

    This function writes the .dst file containing the range and azimuth
    of each receiver to the source. Strictly for the texas data at the moment

    Parameters
    ----------
    receiver_file: str
        csv file with receicer information
    model_file: str
        csv file about the model
    source_coord: np.ndarray of shape (1,3)
        cartesian source coordinates in meters
    dst_file: str
        name of output dst file

    Returns
    -------
    Writes dst_file

    '''

    model = pandas.read_csv(model_file)
    depth_m = model['Depth'].values

    receiver_info = pandas.read_csv(receiver_file)
    easting = receiver_info['Easting'].values
    northing = receiver_info['Northing'].values
    depth = receiver_info['Depth'].values
    group = receiver_info['Group'].values

    # Keep only deepest receivers
    id_kill = np.where(group==2)[0] # id of shallow receivers 1
    id_kill = np.append(id_kill, np.where(group==3)[0]) # id of shallow receivers 2
    id_kill = np.append(id_kill, np.where(group==4)[0]) # id of permantetely dead traces and duplicates

    x_rec = easting - min(easting)
    y_rec = northing - min(northing)
    z_rec = depth + np.abs(min(depth_m))

    x_rec = np.delete(x_rec, id_kill)
    y_rec = np.delete(y_rec, id_kill)
    z_rec = np.delete(z_rec, id_kill)

    # Addtional permantently dead channels to remove
    id_dead = np.array([0, 1, 2, 4, 5, 8, 16, 19, 22, 25, 31, 32, 33, 34, 35, 36, 37, 42, 43, 45, 46, 47, 51, 52, 55, 56, 57, 59, 60, 62, 63, 66, 67, 68, 69, 71, 72, 73, 75, 76, 79, 83, 84, 85, 86, 88, 89, 92, 93, 94, 96, 97, 98, 100, 103, 106, 107, 108, 110, 111, 112, 113, 114, 116, 119, 122, 123, 127, 128, 129, 131, 132, 134, 136, 137, 139, 142, 144, 145, 146, 150, 152, 155, 156, 157, 158, 160, 165, 166, 167, 169, 170, 172, 174, 175, 178, 179, 180, 181, 182, 186, 187, 189, 190, 193])
    x_rec = np.delete(x_rec, id_dead)
    y_rec = np.delete(y_rec, id_dead)
    z_rec = np.delete(z_rec, id_dead)

    receivers = np.append(x_rec, y_rec)
    receivers = np.append(receivers,z_rec)
    receivers = np.reshape(receivers, (3, len(x_rec)))

    rad, azi = cart2polar(receivers, source_coord)
    rad = rad/1000 # Convert to km

    file = open(dst_file, 'w')
    n_rec = len(rad)
    L = ['   {}                               # of distances /distances\n'.format(n_rec)]
    file.writelines(L)

    for i in range(len(rad)):
        L = ['  {:.2f}      {:.2f}\n'.format(rad[i], azi[i])]
        file.writelines(L)

    file.close()

def write_dst2cmd(cmd_file: str, dst_file: str, zs: float):

    """
    write_dst2cmd(cmd_file, dst_file, zs)

    This function writes the path to the dst_file to and the source depth
    to the cmd file.

    Parameters
    ----------
    cmd_file: str
        Path to cmd file
    dst_file: str
        Path to dst file
    zs: float
        cartesian source depth in km

    Returns
    -------
    Writes dst_file

    """
    file=open(cmd_file,'r')
    lines=file.read().splitlines()

    lines[20] = '   {:.4f}                                  Depth of source'.format(zs)
    lines[21] = '"{}"                                    Range and azimuth file'.format(dst_file)

    file.close()

    file = open(cmd_file, 'w')
    file.write('\n'.join(lines))
    file.close()

def MT_components(strike_dip_rake_M0: np.ndarray) -> np.ndarray:

    """
    MT_components(strike, dip, rake, M0)

    This function writes computes the full moment tensor given the strike,
    dip, rake and source moment

    Parameters
    ----------
    strike_dip_rake_M0: np.ndarray of shape (1,4):
        contains strike, dip, rake and scalar seismic moment in that order

    Returns
    -------
    MT: np.ndarray of shape (3,3)
        Full seismic moment tensor

    """

    strike = strike_dip_rake_M0[0,0]
    dip = strike_dip_rake_M0[0,1]
    rake = strike_dip_rake_M0[0,2]
    M0 = strike_dip_rake_M0[0,3]

    M11= -M0*(np.sin(math.radians(dip))*np.cos(math.radians(rake))*np.sin(2*math.radians(strike))
              + np.sin(2*math.radians(dip))*np.sin(math.radians(rake))*(np.sin(math.radians(strike)))**2)

    M12= M0*(np.sin(math.radians(dip))*np.cos(math.radians(rake))*np.cos(2*math.radians(strike))
             + 0.5*np.sin(2*math.radians(dip))*np.sin(math.radians(rake))*np.sin(2*math.radians(strike)))

    M13= -M0*(np.cos(math.radians(dip))*np.cos(math.radians(rake))*np.cos(math.radians(strike))
              + np.cos(2*math.radians(dip))*np.sin(math.radians(rake))*np.sin(math.radians(strike)))

    M22= M0*(np.sin(math.radians(dip))*np.cos(math.radians(rake))*np.sin(2*math.radians(strike))
             - np.sin(2*math.radians(dip))*np.sin(math.radians(rake))*(np.cos(math.radians(strike)))**2)

    M23= -M0*(np.cos(math.radians(dip))*np.cos(math.radians(rake))*np.sin(math.radians(strike))
              - np.cos(2*math.radians(dip))*np.sin(math.radians(rake))*np.cos(math.radians(strike)))

    M33= M0*np.sin(2*math.radians(dip))*np.sin(math.radians(rake))

    M21=M12
    M31=M13
    M32=M23

    MT = np.array([[M11,M12,M13],[M21,M22,M23],[M31,M32,M33]])

    return MT

def write_mod2cmd(cmd_file: str, mod_file: str):

    """
    write_mod2cmd(cmd_file, mod_file)

    This function writes the model file to the cmd file

    Parameters
    ----------
    cmd_file: str
        Path to cmd file
    mod_file: str
        Path to model file

    Returns
    -------
    Modified cmd_file including path to model file

    """

    file = open(cmd_file, 'r')
    lines = file.read().splitlines()
    lines[2] = '"{}"                              Velocity model file'.format(mod_file)
    file.close()

    file=open(cmd_file, 'w')
    file.write('\n'.join(lines))
    file.close()

def writeMT2cmd(cmd_file: str, MT: np.ndarray, dom_freq: float):

    """
    writeMT2cmd(cmd_file, MT)

    This function writes the MT to the cmd file

    Parameters
    ----------
    cmd_file: str
        Path to cmd file
    MT: np.ndarray of shape (3,3)
        Full seimsic moment tensor

    Returns
    -------
    Modified cmd_file including MT

    """

    file = open(cmd_file, 'r')

    lines = file.read().splitlines()

    lines[16] = '   {:.1f}                                  Dominant frequency      [RI]'.format(dom_freq)
    lines[17] = '    {:.2f}      {:.2f}       {:.2f}            Moment tensor Components'.format(MT[0,0],MT[0,1],MT[0,2])
    lines[18] = '    {:.2f}      {:.2f}       {:.2f}'.format(MT[1,0],MT[1,1],MT[1,2])
    lines[19] = '    {:.2f}      {:.2f}       {:.2f}'.format(MT[2,0],MT[2,1],MT[2,2])

    file.close()

    file = open(cmd_file,'w')
    file.write('\n'.join(lines))
    file.close()


def writeErzsolOut2cmd(cmd_file: str, erzsol_seismo_out: str):

    """
    writeErzsolOut2cmd(cmd_file, seisOut)

    This function writes the Path to the erzsol seimogram output to the
    cmd file.

    Parameters
    ----------
    cmd_file: str
        Path to cmd file
    erzsol_seismo_out: str
        Path/name of erzsol3 output file

    Returns
    -------
    Modified cmd_file including path to erzsol3 output file

    """
    file = open(cmd_file, 'r')
    lines = file.read().splitlines()
    file.close()
    lines[1] = '"{}"                                  File for T-X seismogram output'.format(erzsol_seismo_out)
    file = open(cmd_file, 'w')
    file.write('\n'.join(lines))
    file.close()



def write_betaInfo2cmd(
    cmd_file: str,
    strike_dip_rake_M0: np.ndarray,
    source_coord: np.ndarray,
    clusterID: np.ndarray):

    """
    write_betaInfo2cmd(cmd_file, strike_dip_rake_M0, source_coord, clusterID)

    This function writes additional beta information to the end of the
    cmd file.

    Parameters
    ----------
    cmd_file: str
        Path to cmd file
    strike_dip_rake_M0: np.ndarray of shape (1,4)
        contains strike, dip, rake and scalar seismic moment in that order
    source_coord: np.ndarray of shape (1,3)
        cartesian source coordinates in meters
    clusterID: np.ndarray of shape (1,number_of_clusters)
        The one hot vector used for to refer to the cluster.

    Returns
    -------
    Modified cmd_file including strike, dipe rake and seismic moment,
    the cartesian source coorindates and the one_hot_vector

    """

    file = open(cmd_file, 'a')
    file.write('\n\n# strike, dip, rake and seismic moment\n')
    np.savetxt(file, strike_dip_rake_M0, delimiter=' ', fmt='%.0f')
    file.write('\n# Cartesian source coordinates\n')
    np.savetxt(file, source_coord, delimiter=' ', fmt='%.0f')
    file.write('\n# cluster ID vector\n')
    np.savetxt(file, clusterID, delimiter=' ', fmt='%.0f')

    file.close()

def cart2polar(
    cart_receivers: np.ndarray,
    cart_source: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    """
    range, azimuth = cart2polar(cart_receivers, cart_source)

    This function takes in cartesian coordinates of source and receivers
    and returns the distance (range) and azimuth

    Parameters
    ----------
    cart_receivers: np.ndarray of shape (3, number_of_receivers)
        Cartesian coordinates of all receivers
    cart_source: np.ndarray of shape (1,3)
        Cartesian source coordinate
    Inputs:
            cart_receivers, cartesian receiver coordinates (np.array)
            cart_source, cartesian source coordinates (np.array)

    Returns
    -------
    ranges: np.ndarray of shape (number_of_receivers,)
        The source receivers distances/range
    azimuth: np.ndarray of shape (number_of_receivers,)
        The azimuths of all receivers to the source

    """

    ranges = np.zeros(cart_receivers.shape[1])
    azimuth = np.zeros(cart_receivers.shape[1])

    for i in range(cart_receivers.shape[1]):

        ranges[i] = np.sqrt(np.sum((cart_source - cart_receivers[:,i])**2))
        azimuth[i] = np.rad2deg(np.arctan2(cart_receivers[0,i]-cart_source[0,0], cart_receivers[1,i]-cart_source[0,1]))

    return ranges, azimuth



#########################################################
# FUNCTION TO WRITE ERZSOL OUTPUT TO HDF5 DATABASE
# AND TO READ HDF5
#########################################################

def Erzsol3Tohdf5(
    erz_folder: str,
    cmd_folder: str,
    h5_folder: str,
    h5_name: str,
    n_clusters: int,
    ns: int):

    """
    This function writes out a single hdf5 file containing all information
    and the data that can be used for Machine learning and other

    Parameters
    ----------
    erz_folder: str
        Path to folder containing all erzsol data outputs
    cmd_folder: str
        Path to folder containing all erzsol inputs
    h5_folder: str
        Path to folder in whcih to store the h5 file
    h5_name: str
        Name of hdf5 file
    n_clusters: int
        Number of clusters
    ns: int
        Number of time samples per trace (should be equal for all traces)


    Returns
    -------
    Writes the hdf5 file to h5_folder/h5_name

    """

    erzFiles = [f for f in listdir(erz_folder) if f.endswith(".tx.z") if isfile(join(erz_folder, f))]

    filenameH5Output = h5_folder  +'/' + h5_name + '.h5'

    n_examples = len(erzFiles)

    sou_physics = np.zeros((n_examples, 4), dtype='int') # strike, dip, rake, seismic moment
    sou_coordinates = np.zeros((n_examples, 3), dtype='int')
    one_hot_vectors = np.zeros((n_examples, n_clusters), dtype='int')
    azimuths = np.zeros((n_examples, 1))
    ranges = np.zeros((n_examples, 1))


    ncomp = 0
    # Read single file to get number of receivers to initialize data_matrix
    f = open(join(erz_folder,erzFiles[0]), 'rb')
    k = 4
    f.seek(k)
    nt = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
    data_matrix = np.zeros((n_examples, nt, ns))
    f.close()

    # Begin loop over all the input files
    for i, ef in enumerate(erzFiles):

        f = open(join(erz_folder, ef), 'rb')

        cmd_file = cmd_folder + '/' + ef[3:-5] + '.cmd'
        f_cmd = open(cmd_file, 'r')

        lines = f_cmd.read().splitlines()
        f_cmd.close()

        sou_physics[i,:] = np.fromstring(lines[28], dtype='int', sep=' ')
        sou_coordinates[i,:] = np.fromstring(lines[31], dtype='int', sep=' ')
        one_hot_vectors[i,:] = np.fromstring(lines[34], dtype='int', sep=' ')

        # First part information about number of receivers and components per receiver
        k = 4
        f.seek(k)
        n_rec = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
        k+=4
        f.seek(k)
        n_comp = np.fromfile(f,dtype='int32', count=1)[0]   # number of components per receiver
        k+=8

        # Not best prgramming. But does the job. These are the bytes at which to read beta info and data:
        num_bytes = np.array([4,4,5,3,4,4,4,4,4,ns*4,4])

        # Loop over all the receivers and their individual components
        for i_r in range(0, n_rec):
            for j in range(0, n_comp):

                k+=num_bytes[0]
                f.seek(k)
                dist = np.fromfile(f, dtype='float32', count=1)
                k+=num_bytes[1]
                f.seek(k)
                azi = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[2]
                f.seek(k)
                comp = np.fromfile(f,dtype='|S1', count=1).astype(str)[0]
                k+=num_bytes[3]
                f.seek(k)
                dt = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[4]
                f.seek(k)
                ns = np.fromfile(f,dtype='int32', count=1)[0]
                k+=num_bytes[5]
                f.seek(k)
                pcal = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[6]
                f.seek(k)
                tcal = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[7]
                f.seek(k)
                sm = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[8]
                f.seek(k)

                # Only interested in the first component (the vertical component)
                if j == 0:
                    data_matrix[i, i_r, :] = np.fromfile(f, dtype='float32',count=ns)

                k+=num_bytes[9]
                k+=num_bytes[10]

        nt = n_rec
        ncomp = n_comp
        f.close()

    with h5py.File(filenameH5Output, 'w') as hf:

        # Create group and attributes
        g = hf.create_group('Texas synthetic data')
        g.attrs['number of receivers'] = nt
        g.attrs['number of components'] = ncomp
        hf.create_dataset('stike, dip, rake, M0', data=sou_physics, dtype='int32')
        hf.create_dataset('source location', data=sou_coordinates, dtype='int32')
        hf.create_dataset('cluster IDs', data=one_hot_vectors)
        hf.create_dataset('ML array', data=data_matrix, dtype='f')



def read_erzsol3(
    erz_file: str,
    cmd_file: str,
    ns: int)->np.ndarray:

    """
    This function reads a single erz3 file to a npy array

    Parameters
    ----------
    erz_file: str
        File of erzsol data outputs
    cmd_file: str
        File of erzsol inputs
    ns: int
        Number of time samples per trace (should be equal for all traces)


    Returns
    -------
    np.array

    """

    ncomp = 0
    # Read single file to get number of receivers to initialize data_matrix
    f = open(erz_file, 'rb')
    k = 4
    f.seek(k)
    nt = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
    data = np.zeros((nt, ns))
    f.close()

    f = open(erz_file, 'rb')
    f_cmd = open(cmd_file, 'r')

    lines = f_cmd.read().splitlines()
    f_cmd.close()

    # First part information about number of receivers and components per receiver
    k = 4
    f.seek(k)
    n_rec = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
    k+=4
    f.seek(k)
    n_comp = np.fromfile(f,dtype='int32', count=1)[0]   # number of components per receiver
    k+=8

    # Not best prgramming. But does the job. These are the bytes at which to read beta info and data:
    num_bytes = np.array([4,4,5,3,4,4,4,4,4,ns*4,4])

    # Loop over all the receivers and their individual components
    for i_r in range(0, n_rec):
        for j in range(0, n_comp):

            k+=num_bytes[0]
            f.seek(k)
            dist = np.fromfile(f, dtype='float32', count=1)
            k+=num_bytes[1]
            f.seek(k)
            azi = np.fromfile(f,dtype='float32', count=1)
            k+=num_bytes[2]
            f.seek(k)
            comp = np.fromfile(f,dtype='|S1', count=1).astype(str)[0]
            k+=num_bytes[3]
            f.seek(k)
            dt = np.fromfile(f,dtype='float32', count=1)
            k+=num_bytes[4]
            f.seek(k)
            ns = np.fromfile(f,dtype='int32', count=1)[0]
            k+=num_bytes[5]
            f.seek(k)
            pcal = np.fromfile(f,dtype='float32', count=1)
            k+=num_bytes[6]
            f.seek(k)
            tcal = np.fromfile(f,dtype='float32', count=1)
            k+=num_bytes[7]
            f.seek(k)
            sm = np.fromfile(f,dtype='float32', count=1)
            k+=num_bytes[8]
            f.seek(k)

            # Only interested in the first component (the vertical component)
            if j == 0:
                data[i_r, :] = np.fromfile(f, dtype='float32',count=ns)

            k+=num_bytes[9]
            k+=num_bytes[10]

    return np.transpose(data)




def readingHdf5(filename: str) -> np.ndarray:

    """
    dataZ = readingHdf5(filename)

    This function reads an hdf5 file and returns the data

    Parameters
    ----------
    filenameL: str
        hdf5 file

    Returns
    -------
    dataZ: np.ndarray
        The vertical component data
    """

    f = h5py.File(filename, 'r')
    main_key = list(f.keys())[0]

    n_receivers = f[main_key].attrs['number of receivers']
    n_components = f[main_key].attrs['number of components']

    ns = f[main_key][list(f[main_key].keys())[0]].attrs['ns'] # number of samples

    dataList = list(f[main_key])

    #dataR = np.zeros(shape=(ns, n_receivers))
    dataZ = np.zeros(shape=(ns, n_receivers))
    #dataT = np.zeros(shape=(ns, n_receivers))

    cz=0
    #cr=0
    #ct=0
    for i, k in enumerate(list(f[main_key])):

        if k.endswith('R'):
            #dataR[:,cr] = list(f[main_key][k])
            #cr+=1
            pass
        elif k.endswith('Z' ):
            dataZ[:,cz] = list(f[main_key][k])
            cz+=1
        elif k.endswith('T'):
            #dataT[:,ct] = list(f[main_key][k])
            #ct+=1
            pass
        else:
            print('Found other ending')

    return dataZ #dataR, dataZ, dataT

def compute_random_source_locations_per_cluster(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    n_sources_per_cluster: int
    ) -> np.ndarray:

    """

    clusters = compute_random_source_locations_per_cluster(xs, ys, zs, n_sources_per_cluster)

    This function computes random source locations for all the clusters.

    Example
    xMin = 6800, xMax = 7800
    dxC = 90 # spatial x-distance of single cluster
    dx = 10 # remaining distance to reach next cluster
    xs = np.zeros((nx,2))
    for i in range(0,nx):
        xs[i,:] = np.array([x1+i*(dxC+dx), x1+i*(dxC+dx)+dxC])

    Parameters
    ----------
    xs: np.ndarray of shape (spatial clusters in x, 2)
        Contains the xmin and xmax ranges in all x-directions
    ys: np.ndarray of shape (spatial clusters in y, 2)
        Contains the ymin and ymax ranges in all y-directions
    zs: np.ndarray of shape (spatial clusters in z, 2)
        Contains the zmin and zmax ranges in all z-directions

    Returns
    -------
    clusters: np.ndarray of shape (number_of_clusters, 3)
        The cartesian coordinates of all the sources for all clusters

    """
    n_clusters = xs.shape[0]*ys.shape[0]*zs.shape[0]

    # Define random locations of sources within each cluster and assign to one variable

    clusters = np.zeros((n_sources_per_cluster*n_clusters,3))


    # Create n_sources_per_cluster random source locations within each cluster
    c = 0 # Set counter to zero
    for iz in range(0,zs.shape[0]):

        for iy in range(0,ys.shape[0]):

            for ix in range(0,xs.shape[0]):

                pt1 = (xs[ix,0], xs[ix,1])
                pt2 = (ys[iy,0], ys[iy,1])
                pt3 = (zs[iz,0], zs[iz,1])

                # Compute n_sources_per_cluster of random source location in each cluster
                points = [points_in_cube(pt1, pt2, pt3) for _ in range(n_sources_per_cluster)]
                xs_nsou, ys_nsou, zs_nsou = zip(*points)

                clusters[c*n_sources_per_cluster:(c+1)*n_sources_per_cluster,0] = np.array(xs_nsou)
                clusters[c*n_sources_per_cluster:(c+1)*n_sources_per_cluster,1] = np.array(ys_nsou)
                clusters[c*n_sources_per_cluster:(c+1)*n_sources_per_cluster,2] = np.array(zs_nsou)

                c+=1 # Update counter

    return clusters


def points_in_cube(pt1: tuple, pt2:tuple, pt3:tuple)->tuple:

    """
    points_in_cube(pt1, pt2, pt3)

    This function computes random locations in cartesian coordinates between
    the x,y,z ranges defined by pt1, pt2 and pt3, respectively

    Parameters
    ----------
    pt1: tuple
        (xmin, xmax)
    pt2: tuple
        (ymin, ymax)
    pt3: tuple
        (zmin, zmax)

    Returns
    -------
    Tuple (x_random, y_random, z_random) between min max values
    """
    return (random.randrange(pt1[0], pt1[1]+1, 1),random.randrange(pt2[0], pt2[1]+1, 1), random.randrange(pt3[0], pt3[1]+1, 1))


###################################################
# FUNCTIONS FOR PLOTTING
###################################################
def plotClusters(clusters: np.ndarray, n_clusters: int, n_sou: int):

    '''

    plotClusters generates a 3-D scatter plot and colors the clusters
    with random colors in order to distinguish the clusters

    Parameters
    ----------
    clusters: np.ndarray
        All source locations
    n_clusters: int
        Number of clusters
    n_sou: int
        Number of sources per cluster

    Returns
    -------
    3-D scatter plot

    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, n_clusters):
        clr = randomRGB()
        ax.scatter(clusters[i*n_sou:i*n_sou+n_sou,0], clusters[i*n_sou:i*n_sou+n_sou,1], clusters[i*n_sou:i*n_sou+n_sou,2], color=randomRGB())

    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.set_zlabel('Z, m')
    ax.invert_zaxis()
    plt.show()

def randomRGB():
    '''
    This function is called by the plotClusters function.
    It returns a tuple of 3 with random values between 0 and 1. These
    are used to define random RGB-values for the cluster
    '''
    return (random.randrange(0,100,1)/100,random.randrange(0,100,1)/100,random.randrange(0,100,1)/100)


###################################################
# UNUSED/RARELY USED FUNCTIONS
###################################################
def Erzsol3ToMultipleHdf5(erz_folder: str, cmd_folder: str, h5_folder: str):

    '''
    This function writes out a single hdf5 database for each erzsol seimsogram
    output contained in the output folder

    Parameters
    ----------
    erz_folder: str
        Path to folder containing all erzsol data outputs
    cmd_folder: str
        Path to folder containing all erzsol cmd inputs
    h5_folder: str
        Path to folder in whcih to store the h5 file

    Returns
    -------
    hdf5 file


    '''

    #erzFiles = [f for f in listdir(erz_folder) if f!='.DS_Store' if isfile(join(erz_folder, f))]
    erzFiles = [f for f in listdir(erz_folder) if f=='.tx.z' if isfile(join(erz_folder, f))]

    sou_physics = np.zeros((1, 4), dtype='int') # strike, dip, rake, seismic moment
    sou_coordinates = np.zeros((1, 3), dtype='int')
    azimuths = np.zeros((1, 1))
    ranges = np.zeros((1, 1))


    # Begin loop over all the input files
    for i, ef in enumerate(erzFiles):

        f = open(join(erz_folder, ef), 'rb')

        cmd_file = cmd_folder + '/' + ef[3:-5] + '.cmd'
        f_cmd = open(cmd_file, 'r')

        lines = f_cmd.read().splitlines()
        f_cmd.close()

        filenameH5Output = h5_folder + '/h5Out_' + ef[3:-5] + '.h5'

        sou_physics[0,:] = np.fromstring(lines[28], dtype='int', sep=' ')
        sou_coordinates[0,:] = np.fromstring(lines[31], dtype='int', sep=' ')

        # First part information about number of receivers and components per receiver
        k = 4
        f.seek(k)
        n_rec = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
        k+=4
        f.seek(k)
        n_comp = np.fromfile(f,dtype='int32', count=1)[0]   # number of components per receiver
        k+=8

        # Not best prgramming. But does the job. These are the bytes at which to read beta info and data:
        num_bytes = np.array([4,4,5,3,4,4,4,4,4,4096*4,4])

        print(ef)
        # Loop over all the receivers and their individual components
        for i_r in range(0, n_rec):
            for j in range(0, n_comp):

                k+=num_bytes[0]
                f.seek(k)
                dist = np.fromfile(f, dtype='float32', count=1)
                k+=num_bytes[1]
                f.seek(k)
                azi = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[2]
                f.seek(k)
                comp = np.fromfile(f,dtype='|S1', count=1).astype(str)[0]
                k+=num_bytes[3]
                f.seek(k)
                dt = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[4]
                f.seek(k)
                ns = np.fromfile(f,dtype='int32', count=1)[0]
                k+=num_bytes[5]
                f.seek(k)
                pcal = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[6]
                f.seek(k)
                tcal = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[7]
                f.seek(k)
                sm = np.fromfile(f,dtype='float32', count=1)
                k+=num_bytes[8]
                f.seek(k)

                if i == 0 and i_r == 0 and j==0 :
                    data_matrix = np.zeros((ns, n_rec))

                # Only interested in the first component (the vertical component)
                if j == 0:
                    data_matrix[:, i_r] = np.fromfile(f, dtype='float32',count=ns)

                k+=num_bytes[9]
                k+=num_bytes[10]

        f.close()

        with h5py.File(filenameH5Output, 'w') as hf:

            # Create group and attributes
            g = hf.create_group('Texas synthetic data')
            g.attrs['number of receivers'] = n_rec
            g.attrs['number of components'] = n_comp
            hf.create_dataset('stike, dip, rake, M0', data=sou_physics, dtype='int32')
            hf.create_dataset('source location', data=sou_coordinates, dtype='int32')
            hf.create_dataset('data', data=data_matrix, dtype='f')



def write_betaInfo2cmdSingle(cmd_file: str, sdrm: np.ndarray, source_coord: np.ndarray):

    """
    write_betaInfo2cmdSingle(cmd_file, sdrm, source_coord)

    This function writes beta information to the end of the cmd file

    Parameters
    ----------
    cmd_file: str
        Path to cmd file (string)
    sdrm: np.ndarray of shape (1,4)
        contains strike, dip, rake and scalar seismic moment in that order
    source_coord: np.ndarray (1,3)
        Cartesian source coordinates

    Returns
    -------
    Changed cmd_file to include beta info
    """

    file = open(cmd_file, 'a')
    file.write('\n\n# strike, dip, rake and seismic moment\n')
    np.savetxt(file, sdrm, delimiter=' ', fmt='%.0f')
    file.write('\n# Cartesian source coordinates\n')
    np.savetxt(file, source_coord, delimiter=' ', fmt='%.0f')

    file.close()
