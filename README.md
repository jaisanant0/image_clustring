# Image Clustring
The script encodes the faces found in the images and then images with similar face encodings are grouped together in seperate folders.

## Getting started

1. Install the required libraries from requirements.txt
    ```
    pip3 nstall -r requirements.txt
    ```
2. Place all the images to be clustered in one folder.

3. Run 
    ```
    python3 cluster.py
    ```
NOTE : You can edit the the 'model' parameter to [cnn](https://github.com/jaisanant0/image_clustring/blob/9a9f4016510176ecf08274d27a108faaa2ba59df/cluster.py#L30) if you are using GPU.

## Result

For each unique faces detected in images, a seprate folder will be created which will have the images of same people. 
