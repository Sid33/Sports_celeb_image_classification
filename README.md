# Sports_celeb_image_classification
The project aims to classify the uploaded sports celebrity image.

This project has mainly 3 parts: 
  (1) Training part that is in Model folder
  (2) Server part that is a Flask api server , in Server folder
  (3) Webpage that will be a platform for uploading images to classify and for interacting with the flask api

Currently, it is for 5 famous sports players ('lionel messi', 'maria sharapova', 'roger federer', 'serena williams', 'virat kohli').

For the classification process , firstly opencv haarcascade is used to detect the eyes in the image and then wavelet transform is applied to it , finally before going for classification a vertical stack of raw and wavelet transformed is generated. For classification, Grid-SearchCv is used to model selection and parameter tuning.

This project also contains a flask api that accepts the image both in base64 string form and raw image form.

Do place the model file to server folder as well.
