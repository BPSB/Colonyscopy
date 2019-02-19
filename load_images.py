import time
import rawpy
import imageio
import numpy as np
from matplotlib import pyplot as plt
from colonyscopy.tools import smoothen, color_distance, expand_mask, color_sum
from colonyscopy import Colony, Plate, ColonyscopyFailedHeuristic
import os

def load_image(filename):
    raw = rawpy.imread(filename)
    return raw.raw_image_visible.copy()
def show(image):
    plt.clf()
    plt.imshow(image)
    plt.clim(*[np.percentile(image,p) for p in [0.0001,99.9999]])
    plt.colorbar()
    plt.show(block=False)
def load_images_from_folder(foldername):
    x_bound_l = 362
    x_bound_u = 3030
    y_bound_l = 546
    y_bound_u = 4800

    xdim = x_bound_u - x_bound_l
    ydim = y_bound_u - y_bound_l

    # Create a list of filenames in the folder
    names = []
    for root, dirs, files in os.walk(foldername +  '/'):
        for file in files:
            if file.endswith('.CR2'):
                names.append(file)
    names.sort()

    images = np.zeros((len(names),int(xdim/2),int(ydim/2),3))

    for i in range(len(names)):
        im = load_image(foldername + '/' + names[i])
        # Take average over the two green values
        images[i,:,:,0] = (im[x_bound_l:x_bound_u:2,y_bound_l:y_bound_u:2] + im[x_bound_l+1:x_bound_u:2,y_bound_l+1:y_bound_u:2])/2.0
        images[i,:,:,1] = im[x_bound_l+1:x_bound_u:2,y_bound_l:y_bound_u:2]
        images[i,:,:,2] = im[x_bound_l:x_bound_u:2,y_bound_l+1:y_bound_u:2]

    return images
def display_growth_curve(colony):  # New intensity measure
    N_t = np.shape(colony.images)[0]+1
    time = np.linspace(0,(N_t-2)*0.25,N_t-1)
    fit_interval_length = 0.7
    min_lower_bound = 2.0
    bg = np.zeros((3,N_t-1))
    intensity = np.zeros((3,N_t-1))

    pl = np.zeros((3,N_t-1))

    for t in range(N_t-1):
        for m in (0,1,2):
            bg[m,t] = np.sum(np.logical_not(smoothen_mask(Colony1.mask,10)+np.logical_not(Colony1.speckle_mask))*Colony1.images[t,:,:,m])/np.sum(np.logical_not(smoothen_mask(Colony1.mask,10)+np.logical_not(Colony1.speckle_mask)))
            intensity[m,t] = np.sum(np.multiply(Colony1.mask,Colony1.images[t,:,:,m]))/np.sum(Colony1.mask)

    pl = intensity-bg

    if np.min(pl) < 0:
        pl = pl+1.05*abs(np.min(pl))

    pl = np.sum(pl, axis=0)

    smooth_log = savgol_filter(np.log10(pl)[np.logical_not(np.isnan(np.log10(pl)))], 13, 3)
    smooth_time = time[np.logical_not(np.isnan(np.log10(pl)))]
    n_nan = np.sum(np.isnan(np.log10(pl)))

    lower_bound = (np.max(smooth_log)+np.min(smooth_log)-fit_interval_length)/2

    if lower_bound < min_lower_bound:
        lower_bound = min_lower_bound

    upper_bound = lower_bound + fit_interval_length

    for k in range(len(smooth_log)):
        if smooth_log[k] > lower_bound:
            i_0 = k+n_nan
            break


    for k in range(len(smooth_log)):
        if smooth_log[k] > upper_bound:
            i_f = k+n_nan
            break

    a = np.polyfit(time[i_0:i_f], np.log10(pl[i_0:i_f]), 1)

    gen_time = np.log10(2)/a[0]

    print('Calculated generation time in hours is')
    print(gen_time)

    plt.figure(figsize=(12,8))
    plt.plot(time, np.log10(pl), '.', label='Measurement')
    plt.plot(time[i_0:i_f], np.log10(pl[i_0:i_f]), '.', label='Timepoints included in fit')
    plt.plot(time[i_0:i_f], a[0]*time[i_0:i_f] + a[1], label='Fit')
    plt.xlabel('Time [h]')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()
def test_intensity_threshold(colony):
    N_t = np.shape(colony.images)[0]+1
    time = np.linspace(0,(N_t-2)*0.25,N_t-1)
    col_intensity = np.zeros(np.shape(colony.images)[0])

    for t in range(np.shape(colony.images)[0]):
        #col_intensity[t] = np.sum(np.multiply(np.sum(colony.images[t],axis=-1),colony.speckle_mask))/(np.sum(colony.speckle_mask))
        col_intensity[t] = np.sum(np.multiply(color_distance(colony.images[t],colony.background),colony.speckle_mask))/(np.sum(colony.speckle_mask))

    col_intensity = col_intensity-np.min(col_intensity)

    col_smooth = savgol_filter(col_intensity, 15, 3)

    intensity_threshold = 4.5*np.mean(col_smooth[0:30])

    for t in range(np.shape(MyFirstPlate.images)[0]):
        if col_intensity[t] > intensity_threshold:
            print('Intensity threshold of')
            print(intensity_threshold)
            print('is reached at t[h] =')
            print(time[t])
            break

    plt.figure(figsize=(10,8))
    plt.title('Intensity of the whole segment')
    plt.plot(time, col_intensity, label='Raw')
    plt.plot(time, col_smooth, label='Smoothed')
    plt.plot(time, intensity_threshold * np.ones((len(col_intensity))), '--', label='Threshold')
    plt.xlabel('Time [h]')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()

    plt.title('Colony at time t')
    plt.imshow(color_distance(colony.images[t],colony.background))
    plt.show()

    plt.title('Colony at latest captured timepoint')
    plt.imshow(color_distance(colony.images[-1],colony.background))
    plt.show()

    plt.title('Mask of colony created by current mask function')
    plt.imshow(colony.mask)
    plt.show()

    plt.title('Last picture with colony area cut out')
    plt.imshow(np.logical_not(colony.mask)*color_distance(colony.images[-1],colony.background))
    plt.colorbar()
    plt.show()

    plt.title('Colony speckle mask')
    plt.imshow(colony.speckle_mask)
    plt.show()

    plt.title('Pixels that are used for determining the background')
    plt.imshow(np.logical_not(smoothen_mask(colony.mask,10)+np.logical_not(colony.speckle_mask))*color_distance(colony.images[-1],colony.background))
    plt.colorbar()
    plt.show()
def generation_time(Plate): # Old intensity measure
    """
    Returns a matrix which contains the generation times of the colonies on the plate.

    Parameters:

    fit_steps_low: Number of steps that are counted for fit before smoothed curve has reached lower threshold

    fit_steps_up: Number of steps that are not counted for fit before smoothed curve has reached upper threshold

    exp_width: estimated window in ln-scale which the cells grow exponentially

    lag_phase_steps: Number of time steps that the lag phase takes at least + 4 so that the fit can be definitely excluded there

    """
    gen_time = np.zeros((len(Plate.borders[0])-1,len(Plate.borders[1])-1))

    for i in range(len(Plate.borders[0])-1):
        for j in range(len(Plate.borders[1])-1):

            fit_steps_low = 3
            fit_steps_up  = 4
            exp_width = 1.8 #np.log(10)
            lag_phase_steps = 20

            N = np.shape(Plate.images)[0]

            Colony1 = Colony(
                    Plate.images[:,Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1],:],
                    Plate.background[Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1],:],
                    Plate.speckle_mask[Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1]]
                    )

            col_intensity = np.zeros(N)

            time = np.linspace(0,(N-1)*0.25,N)

            for t in range(N):
                col_intensity[t] = Colony1.intensity(t)

            logintensity = np.log(col_intensity)

            col_smooth = savgol_filter(col_intensity, 15, 3)

            z = (np.max(np.log(col_smooth))-np.min(np.log(col_smooth))-exp_width)/2

            for k in range(len(np.log(col_smooth))):
                if (np.log(col_smooth[k]) > np.min(np.log(col_smooth))+z) & (k > lag_phase_steps):
                    i_0 = k-fit_steps_low
                    break

            for k in range(len(np.log(col_smooth))):
                if np.log(col_smooth[k]) > np.max(np.log(col_smooth))-z:
                    i_f = k-fit_steps_up
                    break

            a = np.polyfit(time[i_0:i_f], logintensity[i_0:i_f], 1)

            gen_time[i,j] = np.log(2)/a[0]

            x = len(Colony1.images[0,:,0,0])
            y = len(Colony1.images[0,0,:,0])

            if np.sum(np.logical_not(Colony1.speckle_mask[int(0.25*x):int(0.75*x),int(0.25*y):int(0.75*x)]))/np.size(Colony1.speckle_mask[int(0.25*x):int(0.75*x),int(0.25*y):int(0.75*x)]) > 0.2:
                gen_time[i,j] = None

    gen_time[0,0] = None
    gen_time[0,47] = None

    return gen_time
def generation_time_new(Plate):
    N_t = np.shape(Plate.images)[0]+1
    time = np.linspace(0,(N_t-2)*0.25,N_t-1)
    fit_interval_length = 0.7
    min_lower_bound = 2.0
    gen_time = np.zeros((len(Plate.borders[0])-1,len(Plate.borders[1])-1))

    for i in range(len(Plate.borders[0])-1):
        for j in range(len(Plate.borders[1])-1):

            Colony1 = Colony(
                Plate.images[:,Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1],:],
                Plate.background[Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1],:],
                Plate.speckle_mask[Plate.borders[0][i]:Plate.borders[0][i+1],Plate.borders[1][j]:Plate.borders[1][j+1]]
                )

            bg = np.zeros((3,N_t-1))
            intensity = np.zeros((3,N_t-1))

            for t in range(N_t-1):
                #plt.imshow(np.sum(Colony1.images[t,:,:,:],axis=-1))
                #plt.show()
                for m in (0,1,2,):
                    bg[m,t] = np.sum(np.logical_not(smoothen_mask(Colony1.mask,10)+np.logical_not(Colony1.speckle_mask))*Colony1.images[t,:,:,m])/np.sum(np.logical_not(smoothen_mask(Colony1.mask,10)+np.logical_not(Colony1.speckle_mask)))
                    intensity[m,t] = np.sum(np.multiply(Colony1.mask,Colony1.images[t,:,:,m]))/np.sum(Colony1.mask)

            pl = intensity-bg

            if np.min(pl) < 0:
                pl = pl+1.05*abs(np.min(pl))

            pl = np.sum(pl, axis=0)

            smooth_log = savgol_filter(np.log10(pl)[np.logical_not(np.isnan(np.log10(pl)))], 13, 3)
            smooth_time = time[np.logical_not(np.isnan(np.log10(pl)))]
            n_nan = np.sum(np.isnan(np.log10(pl)))

            lower_bound = (np.max(smooth_log)+np.min(smooth_log)-fit_interval_length)/2

            if lower_bound < min_lower_bound:
                lower_bound = min_lower_bound

            upper_bound = lower_bound + fit_interval_length

            for k in range(len(smooth_log)):
                if smooth_log[k] > lower_bound:
                    i_0 = k+n_nan
                    break


            for k in range(len(smooth_log)):
                if smooth_log[k] > upper_bound:
                    i_f = k+n_nan
                    break

            a = np.polyfit(time[i_0:i_f], np.log10(pl[i_0:i_f]), 1)

            gen_time[i,j] = np.log10(2)/a[0]
    return gen_time

images = load_images_from_folder('plate1')

Plate1 = Plate(images[1:], layout=(30,48), bg = images[0])
Plate1.segment()
print(Plate1.borders)
