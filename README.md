# PGC-Net
A novel deep learning approach for enhancing the accuracy of  image-based cells counting
# Pre-trained models
Pretrained models can be found in our models folder
# data
The data we used in the paper can be found in the data folder
# Usage
## Train
Feel free to use the provided CSV file template for batch training. Here, we offer batch training configuration files in CSV format for four datasets.You can find them in "config_bat" folder.

For example:
`python -u main_bat.py train_VGG_bat.csv output.csv`

The training results will be output in the 'output.csv' file.

## Predict
You can use our trained models or train your own models. Put the model in the models folder and the '.hdf5' file to be tested in the data folder. Run the following code to test

`python -u predict.py predict_config.yml`

In the 'predict_config.yml' file, you can configure the save path for prediction results, the path of the sample data for prediction, the model path, and the scaling factor for model prediction results.

After the test is completed, a predicted result file will be generated in the folders named by index, and a csv file will be generated with the test results of each image.
