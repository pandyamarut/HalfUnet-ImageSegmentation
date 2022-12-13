# HalfUnet-ImageSegmentation

### Implementation Details
- Dataset Used : MICCAI Left Ventricular MRI Dataset - https://sourceforge.net/projects/cardiac-mr/
 -Dataset Description : This dataset contains short-axis images of cardiac MRI scans from multiple cases. There are 45 cases in MICCAI 2009, divided into three groups, and each group contains 15 more cases belonging to different types of failures and the normal scenario. All of the left ventricular MRI cases have endocardium and some have epicardium, so because endocardium is in all of them in this project we only focus on endocardium segmentation.
Among them, 30 cases are used as the training set and 15 cases are used as the test set. The training data came out to be around 550 while the testing was around 260. The data has two divisions, one with all the images in dicom format and the other is a file containing the list of all the contours for creating segmentation masks.

- Dataset Preparation : Data preparation essentially has 3 steps, the first is to read and convert each image from dicom to png format. Second, we created masks from the contour list where in a numpy array all the background pixels are set to 0 while all the foreground pixels are set to 255. Which can be seen in the image below. Also the original structure of this dataset is too complicated and not apt for model training, so we created a directory structure from where we can read data using generators with ease. The image for the directory structure is also attached below.
- Dataset Augmentation : To further diversify the data and get more samples to train on, we augmented data exactly as it was mentioned in the research paper. For each endocardium image we rotate it seven times 45 degrees iteratively and do a horizontal and vertical flip which takes the count to 10 images per origin image. Because we had to control the type of augmentation happening on each image, we used a custom function to do so and not the keras inbuilt one as that does all the augmentation on the go randomly. Finally we end up with around 4900 training images, 500 validation images and 260 testing images.
- Data Preparation and Augmentation code file : data_preparation.ipynb
- Implemented Model Architectures :These are the following models which have been implemented in Keras and further trained on the same data as well as with the same configuration to be able to do a fair comparison between them. The models are as follows :
● Half-UNet : Same as mentioned in [1]
● UNet : Same as mentioned in [3]
● Nested Half-UNet : Model inspired by U^2-Net [4]
● Half-UNet with batch normalization and another with regularization
- Training Configuration and Details : Epochs : 60
Input image size = (256, 256, 3) Batch Size : 2
Learning rate : Initially set to 0.001 and switched to 0.0005 at epoch 30 and 0.0001 at epoch 50. Optimizer : Adaptive moment estimation (Adam)
Loss Function : Binary cross entropy (also tried Dice Loss)
Convolution weight initializer (not for UNet): Kaiming initialization


### Testing Instructions : Steps to be able to test all the saved models on the test images is as follows :
1) In the “DL Project Final.ipynb” file run the first cell which is just the imports and make sure to change the “FOLDER_PATH” to the appropriate base directory containing training, testing and validations dataset and also the training models for testing. The link for the trained models as well as the prepared dataset has been shared in the Data directory.
2) Run the data generator cell with the heading “Data Generator Creation” (second cell)
3) At the end of the python notebook we have a cell with heading “Model loading, evaluation and result visualization” which contains all the required functions to test the
model, we have to run this cell as well.
4) Running the following lines will load the desired model as well as initialize the testing
generator. The names of the models are kept based on the architecture to avoid confusion. “Half_unet_reg” means half_UNet with regularization, similarly “half_unet_norm” means Half_UNet with batch normalization.
5) Finally running the following function will run all the models and plot the outputs, the function can be customized to view runs on different images from the test set by passing start which is the start index and can be between 0-265 (we have 266 test images) and test_img_count which is the number of images to run the models on.
all_models = load_models() # pass any of the following as a string to load the specific model : 'half_unet', 'half_unet_reg', 'half_unet_norm', 'unet', 'nested_half_unet'
gen, num_tests = load_test_data(1) # pass any number higher than 1 which represents the batch_size for the data generator
#to check the output on different data run the below model with start which is the start index and test_img_count which is the total images to test on
#we have total 266 images in the testing set so the start can be anything between 0 and 265 (inclusive) visualize_outputs_of__all_models(gen, all_models, start = 3, test_img_count = 4)





Feel free to add comments/Feedback.
