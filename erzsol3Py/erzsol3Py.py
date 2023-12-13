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
import math
import random
from os import listdir
from os.path import isfile, join
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#########################################################
# FUNCTION USED TO CREATE ERZSOL3 INPUTS
#########################################################
def writeModFile(vp, vs, rho, layers, layer_mode, model_name, erzsol3_mod_file, nr=3, qa=0.0, qb=0.0):

    '''
    model2Erzsol3mod(csv_file_model, erzsol_mod_file)

    This function creates a .mod file which is used for erzsol3.

    Parameters
    ----------
    vp: P-velocities km/s
    vs: S-velocities km/s
    rho: density
    layers: either thickness of each layer or depth of each layer (km)
    layer_mode: if thickness or depth given in layers
        =0 for thickness
        =1 for depth
    model_name: str, name of your model (named inside .mod file)
    erzsol_mod_file: str, name of your .mod file and where you want to save it
    nr: controls number of reverberations in the layer
        nr=0: no reflection from top layer
        nr=1: no internal multiples in layer
        nr>-3: all internal multiples in layer

    For more details check B.L.N Kennet's manual of his ERZSOL3 software.

    Returns
    -------
    erzsol_mod_file

    '''

    if (layer_mode < 0 or layer_mode >1):
        raise Exception("layer_mode not equal to 0 or 1")

    file = open(erzsol3_mod_file, 'w+')

    model_name = model_name + '\n'
    nLayers_type = '    {}        {}\n'.format(len(layers), layer_mode)
    L = [model_name, nLayers_type]
    file.writelines(L)

    for i in range(len(vp)):
        L = ['{}   {:.3f}    {:.3f}     {:.2f}     {:.3f}    {:.3f}     {:.3f}\n'.format(nr, vp[i],vs[i],rho[i],layers[i],qa,qb)]
        file.writelines(L)

    file.close()

def writeDstFile(rxs, sx, dst_file):

    '''
    writeDstFile(rxs, sx, dst_file)

    This function writes the .dst file containing the range and azimuth
    of each receiver to the source. Strictly for the texas data at the moment

    Parameters
    ----------
    rxs: np.ndarray, receiver locations of shape (n_receivers, 3)
        cartesian coordinates in km
    sx: np.ndarray, source location of shape (1,3)
        cartesian coordinates in km
    dst_file: str, name of output dst file

    Returns
    -------
    Writes dst_file

    '''

    if sx.shape != (3,):
        raise Exception(" shape of sx no (3,)")

    if rxs.shape[0] != 3:
        raise Exception("rxs.shape[0] not 3")

    rad, azi = cart2polar(rxs, sx)

    file = open(dst_file, 'w')
    n_rec = len(rad)
    L = ['   {}                               # of distances /distances\n'.format(n_rec)]
    file.writelines(L)

    for i in range(len(rad)):
        L = ['  {:.2f}      {:.2f}\n'.format(rad[i], azi[i])]
        file.writelines(L)

    file.close()


def writeCmdFile(
    cmd_file,
    erz_out_file,
    mod_file,
    dst_file,
    surface_condition,
    ns, dt, MT, source_coord,
    source_center_frequency, low_frequency_taper, high_frequency_taper,
    minimum_slowness, maximum_slowness,
    wavelet="RI", wavelet_file="ew.wav", starttime=0.0,
    n_slowness=10000, slow_plo=10, slow_phi=10, slow_red=0.0,
    exponential_damping="YES", debug_fk="NO", debug_wav="NO"):

    '''
    write cmd file

    parameters
    cmd_file: str, name of cmd file to create
    erz_out_file: str, name of erzsol3 output file (synthetics)
    mod_file: str, .mod file
    dst_file: str, .dst file
    surface_condition: str, surface condition
        (HS, H1, WF, WS)
    ns: int, number of time samples
    dt: float, time step
    MT: np.ndarray, moment tensor (3x3)
    source_coord: float, source depth (km)
    strike_dip_rake_magnitude: strike, dip rake, moment magnitude
    source_center_frequency: float, source center frequency
    low_frequency_taper: float, frequency taper low
    high_frequency_taper: float, frequency taper high
    minimum_slowness: float, minimum slowness
    maximum_slowness: float, maximum slowness

    optional parameters
    wavelet
    wavelet_file
    starttime
    n_slowness
    slow_plo
    slow_phi
    slow_red
    exponential_damping
    debug_fk
    debug_wav

    '''


    file = open(cmd_file, 'w')

    L = ['"ERZSOL3-ew1  "              Title\n']
    L.append('"{}"               File for T-X seismogram output\n'.format(erz_out_file))
    L.append('"{}"                              Velocity model file\n'.format(mod_file))
    L.append('"{}"                                    Surface Condition (HS,H1,WF,WS)\n'.format(surface_condition))
    L.append(' {}                                  Number of slownesses (<2500)\n'.format(n_slowness))
    L.append(' {:.4f}                                Minimum slowness\n'.format(minimum_slowness))
    L.append(' {:.4f}                                Maximum slowness\n'.format(maximum_slowness))
    L.append('{}                                      Slowness taper plo (n samples)\n'.format(slow_plo))
    L.append('{}                                      Slowness taper phi (n samples)\n'.format(slow_phi))
    L.append('"{}"                                    Wavelet input or Ricker (WA/RI)\n'.format(wavelet))
    L.append('"{}"                                Wavelet file\n'.format(wavelet_file))
    L.append('"{}"                                    Exponential damping? (YE/NO)\n'.format(exponential_damping))
    L.append('{}                                  Number of time points\n'.format(ns))
    L.append('  {:.3f}                                   Time step\n'.format(dt))
    L.append('   {:.3f}  {:.3f}                            Frequency taper (low)\n'.format(
        low_frequency_taper[0],
        low_frequency_taper[1]
        )
    )
    L.append('   {:.3f}  {:.3f}                            Frequency taper (high)\n'.format(
        high_frequency_taper[0],
        high_frequency_taper[1]
        )
    )
    L.append('   {:.1f}                                  Dominant frequency      [RI]\n'.format(source_center_frequency))
    L.append('    {:.2f}      {:.2f}       {:.2f}            Moment tensor Components\n'.format(MT[0,0],MT[0,1],MT[0,2]))
    L.append('    {:.2f}      {:.2f}       {:.2f}\n'.format(MT[1,0],MT[1,1],MT[1,2]))
    L.append('    {:.2f}      {:.2f}       {:.2f}\n'.format(MT[2,0],MT[2,1],MT[2,2]))
    L.append('   {:.4f}                                  Depth of source\n'.format(source_coord[2]))
    L.append('"{}"                                    Range and azimuth file\n'.format(dst_file))
    L.append('  {:.1f}                                   Reduction slowness\n'.format(slow_red))
    L.append('  {:.3f}                                 Start time (reduced)\n'.format(starttime))
    L.append('"{}"                                    Debug/frequency-wavenumber (YE/NO)\n'.format(debug_fk))
    L.append('"{}"                                    Debug/waveform (YE/NO)\n'.format(debug_wav))

    '''
    L.append('\n\n# strike, dip, rake and seismic moment:\n')
    L.append('{} {} {} {} \n'.format(
        strike_dip_rake_magnitude[0,0],
        strike_dip_rake_magnitude[0,1],
        strike_dip_rake_magnitude[0,2],
        strike_dip_rake_magnitude[0,3]
        )
    )
    '''

    L.append('# Cartesian source coordinates\n')
    L.append('{} {} {}'.format(source_coord[0], source_coord[1], source_coord[2]))

    file.writelines(L)
    file.close()

    return None

def MT_components(strike_dip_rake_M0):

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


def cart2polar(
    cart_receivers,
    cart_source
):

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
        azimuth[i] = np.rad2deg(np.arctan2(cart_receivers[0,i]-cart_source[0],
                                           cart_receivers[1,i]-cart_source[1]))

    return ranges, azimuth



#########################################################
# READ ERZSOL3 OUPUT TO NPY ARRAY
#########################################################

def readErzsol3(erz_file, cmd_file):

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

    # Read number of time-samples from cmd file
    f_cmd = open(cmd_file, 'r')
    lines = f_cmd.read().splitlines()
    f_cmd.close()
    ns = int(lines[12].split(' ')[0]) # Get number of time-samples

    # Read single file to get number of receivers to initialize data_matrix
    f = open(erz_file, 'rb')
    k = 4
    f.seek(k)
    nt = np.fromfile(f,dtype='int32', count=1)[0]  # number of receivers
    data = np.zeros((3, nt, ns))
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

    # Not best programming. But does the job. These are the bytes at which to read beta info and data:
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
            # change two three components
            data[j, i_r, :] = np.fromfile(f, dtype='float32',count=ns)
            #if j == 0:
            #    data[i_r, :] = np.fromfile(f, dtype='float32',count=ns)

            k+=num_bytes[9]
            k+=num_bytes[10]

    return data


def compute_random_source_locations_per_cluster(
    xs,
    ys,
    zs,
    n_sources_per_cluster
    ):

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


def points_in_cube(pt1, pt2, pt3):

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

import copy
import numpy as np
def wiggle(
    DataO,
    x=None,
    t=None,
    skipt=1,
    lwidth=.5,
    gain=1,
    typeD='VA',
    color='red',
    perc=100):

    """
    wiggle(DataO, x=None, t=None, maxval=-1, skipt=1, lwidth=.5, gain=1, typeD='VA', color='red', perc=100)

    This function generates a wiggle plot of the seismic data.

    Parameters
    ----------
    DataO: np.ndarray of shape (# time samples, # traces)
        Seismic data

    Optional parameters
    -------------------
    x: np.ndarray of shape Data.shape[1]
        x-coordinates to Plot
    t: np.ndarray of shape Data.shap[0]
        t-axis to plot
    skipt: int
        Skip trace, skips every n-th trace
    ldwidth: float
        line width of the traces in the figure, increase or decreases the traces width
    typeD: string
        With or without filling positive amplitudes. Use type=None for no filling
    color: string
        Color of the traces
    perc: float
        nth parcintile to be clipped

    Returns
    -------
    Seismic wiggle plot

    Adapted from segypy (Thomas Mejer Hansen, https://github.com/cultpenguin/segypy/blob/master/segypy/segypy.py)


    """
    # Make a copy of the original, so that it won't change the original one ouside the scope of the function
    Data = copy.copy(DataO)

    # calculate value of nth-percentile, when perc = 100, data won't be clipped.
    nth_percentile = np.abs(np.percentile(Data, perc))

    # clip data to the value of nth-percentile
    Data = np.clip(Data, a_min=-nth_percentile, a_max = nth_percentile)

    ns = Data.shape[0]
    ntraces = Data.shape[1]

    fig = plt.gca()
    ax = plt.gca()
    ntmax=1e+9 # used to be optinal

    if ntmax<ntraces:
        skipt=int(np.floor(ntraces/ntmax))
        if skipt<1:
                skipt=1

    if x is not None:
        x=x
        ax.set_xlabel('Distance [m]')
    else:
        x=range(0, ntraces)
        ax.set_xlabel('Trace number')

    if t is not None:
        t=t
        yl='Time [s]'
    else:
        t=np.arange(0, ns)
        yl='Sample number'

    dx = x[1]-x[0]

    Dmax = np.nanmax(Data)
    maxval = np.abs(Dmax)

    for i in range(0, ntraces, skipt):

       # use copy to avoid truncating the data
        trace = copy.copy(Data[:, i])
        trace = Data[:, i]
        trace[0] = 0
        trace[-1] = 0
        traceplt = x[i] + gain * skipt * dx * trace / maxval
        traceplt = np.clip(traceplt, a_min=x[i]-dx, a_max=(dx+x[i]))

        ax.plot(traceplt, t, color=color, linewidth=lwidth)

        offset = x[i]

        if typeD=='VA':
            for a in range(len(trace)):
                if (trace[a] < 0):
                    trace[a] = 0
            ax.fill_betweenx(t, offset, traceplt, where=(traceplt>offset), interpolate='True', linewidth=0, color=color)
            ax.grid(False)

        ax.set_xlim([x[0]-1, x[-1]+1])

    ax.invert_yaxis()
    ax.set_ylim([np.max(t), np.min(t)])
    ax.set_ylabel(yl)


def plotClusters(clusters, n_clusters, n_sou):

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
