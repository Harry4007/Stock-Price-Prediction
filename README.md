# Stock-Price-Prediction
In this project I build a Stock price prediction using LSTM.

Pipeline for building a stock price prediction using LSTM from scratch:

### Data Acquisition: 
Obtain the historical stock price data for the desired stock symbol. For this, I uses libraries like pandas_datareader to fetch the data from online sources like Yahoo Finance.

### Data Preprocessing: 
Preprocess the data to make it suitable for training the LSTM model. This includes steps like:

● Extracting the relevant features, such as the closing price (I only uses Closing price for this model).

● Scaling the data to a specific range (e.g., between 0 and 1) using MinMaxScaler.

● Splitting the data into training and testing sets.

### LSTM Model Architecture: 
This involves specifying the number of LSTM layers, the number of units in each layer, and any additional layers like Dense layers. 

### Model Compilation: 
Compile the LSTM model by specifying the optimizer and loss function. Typically, Adam optimizer and mean squared error (MSE) loss are used for regression tasks.

### Model Training: 
Now comes the Training part. Train the LSTM model using the training dataset. This involves fitting the model to the training data, specifying the number of epochs (iterations over the entire training data) and batch size (the number of samples used in each training iteration).

### Model Evaluation: 
Evaluate the trained LSTM model using the testing dataset. This involves making predictions on the test data and comparing them with the actual values to calculate metrics such as mean squared error (MSE) or root mean squared error (RMSE).

### Prediction: 
After evaluating the model, you can use it to make predictions on new, unseen data. This involves preprocessing the new data in the same way as the training data and using the trained model to predict the future stock prices.

### Visualization: 
Visualize the predicted stock prices along with the actual prices to analyze the model's performance. You can use libraries like matplotlib to plot the data and observe the trends.
