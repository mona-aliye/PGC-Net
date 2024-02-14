# PGC-Net
A novel deep learning approach for enhancing the accuracy of image-based cells counting.
# Pre-trained models
You can download our trained model in the link below：

[models](https://drive.google.com/file/d/1hoBHjqnpTY6HbjCy8QiYdaA8ylgFR4av/view?usp=drive_link)

Pretrained models should be put in models folder.

# data
You can download datasets in the link below：

[datasets](https://drive.google.com/drive/folders/1P1xxG5F8-DeaCuXRh4dN52N3YiW8LsBz?usp=drive_link)

Datasets should be put in data folder.

# Usage
## Train
Feel free to use the provided CSV file template for batch training. Here, we offer batch training configuration files in CSV format for four datasets. Then run the following code to train


`python -u main_bat.py train_VGG_bat.csv output.csv`

The training results will be output in the 'output.csv' file.

## Predict
You can use our trained models or train your own models. Put the model in the models folder and the '.hdf5' file to be predicted in the data folder. Run the following code to predict

`python -u predict.py predict_config.yml`

In the 'predict_config.yml' file, you can configure the save path for prediction results, the path of the sample data for prediction, the model path, and the scaling factor for model prediction results.

After completing the predicting, each prediction result file will be generated in a folder named after its index.
