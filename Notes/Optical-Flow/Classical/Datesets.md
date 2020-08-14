### Testing Optical Flow 

- Nothing can be confirmed without a good performance score in the standard datasets. So here are the widely accepted datasets for testing Optical Flow algoruthms :

## Datasets
- [`Middlebury`](http://vision.middlebury.edu/flow/) `2009` [`paper`](http://vision.middlebury.edu/flow/floweval-ijcv2011.pdf)
  - 8 image pairs for training, with ground truth flows generated using four different techniques
  - Displacements are very small, typi- cally below 10 pixels.
- [`KITTI`](http://www.cvlibs.net/datasets/kitti/) `2012` [`paper`](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)
  - 194 training image pairs, large displacements, contains only a very special motion type
  - The ground truth is obtained from real world scenes by simultaneously recording the scenes with a camera and a 3D laser scanner.
  - Task: stereo, flow, sceneflow, depth, odometry, object, road, tracking, semantics, etc.
- [`MPI Sintel`](http://sintel.is.tue.mpg.de/) `2012` [`paper`](http://files.is.tue.mpg.de/black/papers/ButlerECCV2012-corrected.pdf)
  - 1041 training image pairs, ground truth from rendered artificial scenes with special attention to realistic image properties
  - Very long sequences, large motions, specular reflections, motion blur, defocus blur, atmospheric effects
  - Task: optical flow.
- [`Flying Chairs (Vision group, Uni-Freiburg)`](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) `2015` [`paper`](https://arxiv.org/abs/1504.06852)
  - 22872 image pairs, a synthetic dataset with optical flow ground truth
  - Task: optical flow.
- [`ChairsSDHom (Vision group, Uni-Freiburg)`](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) `2017` [`paper`](https://arxiv.org/abs/1612.01925)
  - Task: optical flow
  - Designed to be robust to untextured regions and to produce flow magnitude histograms close to those of the UCF101 dataset (small displacement, less than 1 pixel).


### Performance and Accuracy analysis







###### Example

- Original Parameters 


| alpha | pyramid scale | min width | outer iter | inner iter | sor iter | RGB/Gray |
|-------|---------------|-----------|------------|------------|----------|----------|
| 0.011 |      0.75     |     20    |     7      |     1      |    30    |    0     |


Finished Processing
Time Taken: 7.39 seconds for image of size (480, 640, 3)



| Middlebury dataset | Average AE |   R1.0   |   R3.0   |   R5.0   |   A50   |   A75   |   A95    |
|--------------------|------------|----------|----------|----------|---------|---------|----------|
|       Urban2       |  2.788123  | 41.16569 | 16.96549 |  9.8418  | 0.63827 | 2.05376 | 10.42582 |
|    RubberWhale     |  4.587802  | 59.93093 | 19.80042 | 12.9125  | 1.23051 | 2.33691 | 18.49122 |
|       Grove3       |  6.124448  | 60.62402 | 29.27897 | 20.40007 | 1.38861 |  3.7717 | 25.84034 |
|     Dimetrodon     | 3.5564358  | 79.54499 | 44.42406 | 23.72903 | 2.60643 | 4.84857 | 9.79722  |
|     Hydrangea      | 2.3159814  | 29.29121 | 17.81146 | 11.78299 | 0.60361 | 1.59563 | 10.48321 |
|       Urban3       |  7.509062  | 31.05208 | 20.24674 | 19.07747 | 0.52039 | 1.41858 | 50.86626 |
|       Venus        | 7.3363495  | 89.0614  | 40.15664 | 18.52005 | 2.43019 | 3.77595 | 13.1407  |
|       Grove2       | 2.3949218  | 47.09538 | 10.45215 | 6.87109  | 0.94398 | 1.58859 |  7.9697  |