# **Vehicle Detection Project**
### a Self-Driving Car Engineer Nanodegree project at Udacity

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[hog1_ex]: ./examples/HOG_example1.jpg
[hog2_ex]: ./examples/HOG_example2.jpg
[hog3_ex]: ./examples/HOG_example3.jpg
[20_area_box_heat]: ./examples/scale20_area_boxes_heatmap.jpg
[125_area_box_heat]: ./examples/scale125_area_boxes_heatmap.jpg
[box_heat_label]: ./examples/boxes_heatmap_labels.jpg
[x6_box_heat_label]: ./examples/x6_boxes_heatmap_labels.jpg
[sliding_windows]: ./examples/sliding_windows.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
In the ´single_img_features(...)´ function I convert to YCrCb colorspace and iterate all color channels, sending them one by one to ´get_hog_features(...)´. Features returned is appended in an array that, when completed, will contain spatial, histogram and HOG features for the image (Code lines *** TODO ***). The ´get_hog_features(...)´, extract feature data from ´skimage.feature.hog(...)´ and return them (Code lines *** TODO ***).

Parameters to ´hog(...)´ was set to:

|Parameter|Value|
|---------|-----|
|orientations|9|
|pixels_per_cell|8|
|cells_per_block|2|
|transform_sqrt|False|

Examples of HOG visualization on car and not car images:

![HOG example 1][hog1_ex]
![HOG example 2][hog2_ex]
![HOG example 3][hog3_ex]



#### 2. Explain how you settled on your final choice of HOG parameters.

With the well known trial and error technique, varouis combinations was tested, before I settled for the YCrCb colorspace and the parameters seen above.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial , histogram and HOG features (Code lines *** TODO ***). The trainingdata was shuffled, and 10 % of it was split to be testdata. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

With trial and error, again, I decided to search with two scales and an area for each scale. Scales used are 2.0 and 1.25. Overlap is 2 cells, which equals 75 % overlap. (Code lines *** TODO ***)

Example of 2.0 scale search:
![search area for 2.0 scale][20_area_box_heat]

Example of 1.25 scale search:
![search area for 1.25 scale][125_area_box_heat]

To visualize the sliding window grid, this image shows one sliding window in every row of windows in each search area.
![sliding window example][sliding_windows]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![box, heat and label example][box_heat_label]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I save the boxes with positive detections from the last 20 frames. The solution(at the moment) are using boxes from the last 5 frames to create a heatmap. The heatmap is treshholded. Pixel values of 3 and less is considered false positive and is set to zero. The heatmap is then used with `scipy.ndimage.measurements.label()` to identify individual blobs. Every blob is assumed to be a vehicle.

When processing single images, no history of positive hits is available and thresholding heatmap is not effective
Here are examples showing boxes, corresponding heatmap, and labels found. (No threshhold on heatmap is applied):


![x6 box, heat and labels example][x6_box_heat_label]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I feel the solution is pretty robost in its simpleness.

Identifying big or high vehicles in close range may not work at all. 

There are some interesting improvements I´d like to try. Combining with P4 and focus search areas where the road is known to be and track identified vehicles. Reducing the number of sliding windows by less overlap and focus on identified cars. Distance to other vehicles. Make the label-box be more stable, not changing in size so much between frames.

I guess this writeup can be improved also, but prio is term 2 :)
