# Assignment 2 - Digital Image Processing

This repository contains the code and results for Assignment 2 of the Digital Image Processing course at IISc.

## Table of Contents
- [Introduction](#introduction)
- [Instructions](#instructions)
- [Questions](#questions)
    - [Question 1: Spatial Filtering and Binarisation](#question-1-spatial-filtering-and-binarisation)
    - [Question 2: Fractional Scaling with Interpolation](#question-2-fractional-scaling-with-interpolation)
    - [Question 3: Photoshop Feature](#question-3-photoshop-feature)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this assignment, we explore various concepts in digital image processing. We implement functions to perform spatial filtering, binarisation, fractional scaling with interpolation, and a Photoshop-like brightness/contrast adjustment feature.

## Instructions
- Clone this repository to your local machine.
- Install the required libraries mentioned in the code.
- Run the code files to obtain the results.
- Refer to the PDF report for detailed explanations, images, and inferences.

## Questions
### Question 1: Spatial Filtering and Binarisation
In this question, we apply Gaussian blurring on the image 'moon noisy.png' using a spatial Gaussian filter. We generate the filter kernel and convolve it with the input image to obtain the blurred image. Then, we apply Otsu's Binarization algorithm on the blurred image and find the optimal within-class variance for different blur parameters. We plot the histogram and the binarized image for each blur parameter and analyze the results.

### Question 2: Fractional Scaling with Interpolation
In this question, we downsample the image 'flowers.png' by 2 and then upsample the result by 3 using bilinear interpolation. We also upsample the image by 1.5 using bilinear interpolation. We compare the results and provide comments on the differences observed.

### Question 3: Photoshop Feature
In this question, we implement the Brightness/Contrast feature in Adobe Photoshop based on our knowledge of pointwise operations applied to images. We implement two functions, `brightnessAdjust(img, p)` and `contrastAdjust(img, p)`, that adjust the brightness and contrast of the input image based on the parameter `p`. We test our implementation on the image 'brightness contrast.jpg' and analyze the results.

## Results
The results, including images, histograms, and inferences, can be found in the PDF report submitted along with the code files.

## Conclusion
In this assignment, we explored various techniques in digital image processing, including spatial filtering, binarisation, fractional scaling with interpolation, and implementing a Photoshop-like brightness/contrast adjustment feature. We obtained interesting results and made observations based on the experiments conducted.

## Contributing
Contributions to this repository are welcome. If you find any issues or have suggestions for improvement, please create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).
