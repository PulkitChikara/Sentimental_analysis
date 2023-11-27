# Sentimental_analysis
This is the sentimental Analysis of IMDB reviews.

# Model Architecture:
Transformer Model for Sentiment Analysis
The implemented model utilizes a Transformer architecture for sentiment analysis. The key components include:
   Embedding Layer: Converts input tokens into dense vectors.
   Transformer Block: Utilizes multi-head self-attention and feedforward layers for capturing contextual information.
   Global Average Pooling: Aggregates information across the sequence.
   Dense Layers: Two dense layers with ReLU activation for further feature extraction.
   Output Layer: A single neuron with a sigmoid activation function for binary sentiment classification.
   
This architecture leverages the self-attention mechanism in Transformers, allowing the model to capture long-range dependencies in the input sequences efficiently.

# Choice of Dataset:
The model is trained and evaluated on the IMDb Reviews dataset obtained from TensorFlow Datasets. This dataset contains movie reviews labeled with sentiment (positive or negative). The choice of this dataset is motivated by its suitability for binary sentiment analysis tasks and its availability in TensorFlow Datasets.

# Challenges Faced:
During the implementation of the model, several challenges were encountered and addressed:
Model Tuning: Experimentation with hyperparameters and model architecture to achieve optimal performance.
Training Time: Transformers can be computationally expensive to train, requiring careful consideration of resources.
Data Preprocessing: Handling and preprocessing textual data, including tokenization and padding.
Learning Rate Scheduling: Implementing an effective learning rate schedule for improved training convergence.

# Model Evaluation:
After training the model, the following performance metrics were achieved on the testing set:

Accuracy:  51%

Precision:  51%

Recall:  48%

F1 Score:  49%

The confusion matrix provides additional insights into the model's performance, visualizing true positives, true negatives, false positives, and false negatives.
The accuracy and precision are less due to the dataset being huge and the dataset used is from the TensorFlow library itself hence the accuracy is less which I will increase.
