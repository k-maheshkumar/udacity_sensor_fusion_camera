# Results

1. With crossCheck OFF, there are more number of matches with some wrong correspondences. WIth crossCheck ON, the number of matches appears to be comparatively smaller and also has no wrong correspondences 

2. knnMatch
    <!-- - when crossCheck is false, the computation is slower than when crossCheck is true  -->
    - crossCheck = false
        - BRISK_large: about 44% of matches were discarded
        - BRISK_small: about 45% of matches were discarded
    - crossCheck = true
        - BRISK_large: about 45% of matches were discarded
        - BRISK_small: about 41% of matches were discarded

3. BF matching and FLANN matching on the 'BRISK_large' dataset and on the SIFT dataset
    - In less time BF matching on BRISK_large dataset with crossCheck = false, produces same output as FLANN matching on BRISK_large dataset with NN and KNN (k=1)
    - In less time BF matching on SIFT dataset  with crossCheck = false, produces same output as FLANN matching on SIFT dataset with NN and KNN (k=1)
    - On SIFT dataset with KNN(k=2), in less time BF matching has same discard percentage (about 45 %) as FLANN matching. 
    - Also BF matching on BRISK_large dataset with KNN (k=2)has same discard percentage (45 %) as above, while FLANN matching on BRISK_large dataset KNN (k=2) has more discard percentage (about 59 %)