1. Implemented a function to match current bounding boxes with previous bounding boxes based on the max number of keypoints. Matches are exclusive.

2. Lidar TTC was implemented based on the median lidar point value.

3. Implemented a function to only include the keypoint matches where both keypoints (crrent and previous) are within the current bounding box.

4. Camera TTC was implemented based on the median distance ratio from all the keypoints like in the lessons.

5. If I use BRISK keypoints or ORB descriptors, then all TTC values seem pretty reasonable for the Lidar TTC.  Looking from above I don't see anything abnormal. I think my filter method and matching works really well to discard spurious lidar points.

6. If I use BRISK keypoints and ORB descriptors, then all TTC values seem pretty reasonable. But if I use ORB keypoints with ORB descriptors then some of the TTC values are obviousy wrong (very large values).  I think this occurs because the bounding box contains some of the car ahead of it as well and maybe some keypoints are not matches well.  My filter to reject spurious keypoints could be better.  I only tested BRISK+BRISK, ORB+ORB, and BRISK+ORB. The latter had the best results. Average TTC was around 12.8 seconds over all the images for BRISK+ORB and averaging Lidar and Camera TTCs.
(Excel spreadsheet attached with some graphs and data for the detector/descriptor combos tested)
