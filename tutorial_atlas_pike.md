# Tutorial - Improving your data collection

## Train an AtlasPike model

After you have collected a few grid squares you can use the gathered information to improve your data collection. CryoCrane houses a third program which is called AtlasPike. This is a neural-network that you can train with your screening data. Around each exposure AtlasPike extracts patches from the atlas image and associates them with the predicted scores generating a training data set. You can either train a model on the CryoPike score, the NyquistPike score or a combination of both.

You can then specify training parameters and let AtlasPike predict the expected score based on the atlas image. A good model will have a mean absolute error for the validation data set below 0.1. For training the following parameters were usually quite good. 

|Parameter | Value | 
| ------- | ------- |
|Model | ResNet8 | 
|Patch size | 32 | 
|dropout | 0.2 | 
|score | combined | 
|learning rate | 1e-4 | 
|epochs | 100 |
|name | fidibus |

If the specified number of epochs is reached, the training will stop. If you see that the training does not yield any improvement, you can click "Stop training". CryoCrane will save the training state with the lowest validation loss in the folder /atlas_weights. If you are satisfied with your training, click "Stop training and close dialog". 

## Use an AtlasPike model

To run the model on your atlas select your model from the dropdown menu. The models are named by the following convention: 
```
Model_Patchsize_dropout_score_name.pth
```
The model trained with the parameters above would be saved as:
```
ResNet8_32_0.2_combined_fidibus.pth
```

Then click the "Predict atlas" button. In the background the atlas images wil be converted to overlapping patches matching the specified patch size. For each patch AtlasPike will predict a score. The progress of the prediction will be displayed on the progress bar. With a patch size of 32 the model needs around 5 minutes to evaluate the atlas. After the prediction has finished a new option will appear in the "colour by" dropdown menu named: "prediction heat-map". Furthermore, the squares with highest score will appear with a white outline and a number. The number of squares to show can be varied by specifying the number in the box XXX.  

## Select grid squares

Each grid square is associated with its mean brightness, its area and the highest predicted score within. You can update your grid selection by specifying the ranges for the three parameters in these spin boxes: 

XXX Insert image XXX