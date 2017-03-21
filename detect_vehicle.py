import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip

import glob
import os

from scipy.ndimage.measurements import label

BASE_PATH_VEHICLES = 'vehicles/'
BASE_PATH_NON_VEHICLES = 'non-vehicles/'

sub_dirs = os.listdir(BASE_PATH_VEHICLES)
cars = []
for dir in sub_dirs:
	cars.extend(glob.glob(BASE_PATH_VEHICLES + dir + '/*'))

sub_dirs = os.listdir(BASE_PATH_NON_VEHICLES)
non_cars = []
for dir in sub_dirs:
	non_cars.extend(glob.glob(BASE_PATH_NON_VEHICLES + dir +'/*'))

print('number of cars: %i' % len(cars))
print('number of NON-cars: %i' % len(non_cars))


class BoxMemory():
    '''Class for storing boxes from up to 20 frames
    '''
    def __init__(self):
        # box arrays from former frames
        self.recent_boxes = [] 

    def add_boxes(self, boxes):
    	self.recent_boxes.insert(0, (boxes))

    	if (len(self.recent_boxes) > 20):
    		self.recent_boxes = self.recent_boxes[:20]


    def get_boxes_from_X_frames(self, frames = 5):
        len_recent_frames = len(self.recent_boxes)

        if len_recent_frames < frames:
            frames = len_recent_frames

        x_boxes = self.recent_boxes[:frames]

        return x_boxes


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	draw_img = np.copy(img)

	for box in bboxes:
		cv2.rectangle(draw_img, box[0], box[1], color, thick)

	return draw_img


def color_hist(img, nbins=32):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins)
    ghist = np.histogram(img[:,:,1], bins=nbins)
    bhist = np.histogram(img[:,:,2], bins=nbins)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()

    return np.hstack((color1, color2, color3))


def plot3d(pixels, colors_rgb,
        axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)

        features.append(single_img_features(image, color_space, spatial_size,
        	hist_bins, orient,
        	pix_per_cell, cell_per_block, hog_channel,
        	spatial_feat, hist_feat, hog_feat))

    return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
        	if vis == True:
        		hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient,
	            	pix_per_cell, cell_per_block, vis=True, feature_vec=True)

        	else:
	            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
	            	pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis == True:
    	return np.concatenate(img_features), hog_image
    else:
    	return np.concatenate(img_features)

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)

    return img


def visualize(fig, rows, cols, imgs, titles):
	for i, img in enumerate(imgs):
		plt.subplot(rows, cols, i+1, xticks=[], yticks=[])
		plt.title(i+1)
		img_dims = len(img.shape)
		if img_dims < 3:
			plt.imshow(img, cmap='hot')
			plt.title(titles[i])
		else:
			plt.imshow(img)
			plt.title(titles[i])

	# plt.savefig('output_images/boxes_heatmap_labels.jpg')

  

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def find_cars(img, search_boxes, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    # heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255
    img_boxes = []
    
    search_window_count = 0

    # for scale, ystart, ystop in zip(scales, ystarts, ystops):
    for scale, box in zip(scales, search_boxes):
        xstart, ystart = box[0]
        xstop, ystop = box[1]

        img_tosearch = img[ ystart:ystop , xstart:xstop ,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        search_window_count += nxsteps * nysteps

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left+xstart, ytop_draw+ystart),(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)                    
                    img_boxes.append(((xbox_left+xstart, ytop_draw+ystart),(xbox_left+xstart+win_draw,ytop_draw+win_draw+ystart)))


    return draw_img, img_boxes


def setup_SVC(color_space, spatial_size, hist_bins, orient, pix_per_cell,
    cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat):

    t = time.time()

    test_cars = cars
    test_notcars = non_cars

    print('Extracting features...')
    cars_features = extract_features(test_cars, color_space=color_space, spatial_size=spatial_size,
    	hist_bins=hist_bins, orient=orient,
    	pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
    	spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    notcars_features = extract_features(test_notcars, color_space=color_space, spatial_size=spatial_size,
    	hist_bins=hist_bins, orient=orient,
    	pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
    	spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

    print(round(time.time()-t, 2), 'Sec to extract features on dataset.')

    X = np.vstack((cars_features, notcars_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))


    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.1, random_state=rand_state)

    print('Using:\n', orient, 'orientations,\n', pix_per_cell, 'pixels per cell,\n',cell_per_block,'cells per block,\n',
    	hist_bins,'histogram bins, and\n', spatial_size, 'spatial sampling')
    print('feature vector length:', len(X_train[0]))

    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'Sec to train SVC.')
    print('Test accuracy of SCV = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler


def process_image_array(imgArr):
    out_titles = []
    out_images = []

    for imgPath in imgArr:
        img = mpimg.imread(imgPath)
        heatmap = np.zeros_like(img[:,:,0])

        img_with_boxes, out_boxes = find_cars(img, search_boxes, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        heatmap = add_heat(heatmap, out_boxes)

        labels = label(heatmap)
        img_with_labels = draw_labeled_bboxes(np.copy(img), labels)

        out_images.append(img_with_boxes)
        out_titles.append('sliding windows example')
        out_images.append(heatmap)
        out_titles.append('heatmap')
        out_images.append(img_with_labels)
        out_titles.append('labels')

    fig = plt.figure(figsize=(12,32)) # 6 x 2 -> 12, 32
    visualize(fig, 6, 3, out_images, out_titles)





def image_pipeline(img):
	heatmap = np.zeros_like(img[:,:,0])
	    
	out_img, out_boxes = find_cars(img, search_boxes, scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	box_memory.add_boxes(out_boxes)
	out_boxes = box_memory.get_boxes_from_X_frames(frames = 8)

	for box in out_boxes:
		heatmap = add_heat(heatmap, box)

	heatmap = apply_threshold(heatmap, 6)

	labels = label(heatmap)

	draw_img = draw_labeled_bboxes(np.copy(img), labels)

	return draw_img



color_space = 'YCrCb' # RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9 # usually between ? to ?
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # 0, 1, 2, 'ALL'
spatial_size = (32, 32) # Spatial binning dimansions
hist_bins = 32 # Number of histogram bins
spatial_feat = True # Spatial feats on or off
hist_feat = True # Histogram feats on or off
hog_feat = True # HOG feats on or off

svc, X_scaler = setup_SVC(color_space=color_space, spatial_size=spatial_size,
        hist_bins=hist_bins, orient=orient,
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)




box_memory = BoxMemory()
    
scales = [2.0, 1.25]

search_boxes = [[(200,350),(1280,720)],
                [(500,380),(1080,520)]]



challenge_output = 'output_images/vehicle_detection_video.mp4'
clip2 = VideoFileClip('project_video.mp4')
challenge_clip = clip2.fl_image(image_pipeline)
challenge_clip.write_videofile(challenge_output, audio=False)


# imgArr = glob.glob('test_images/test*.jpg')
# process_image_array(imgArr)
# plt.show(block=True)