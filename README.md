# deep-learning-challenge



Overview of the analysis: 

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. 
Leveraging machine learning and neural networks, we use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
Our data is a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

Data Preprocessing
Data Clean up - columns "EIN" (identifier ID) and Name should be removed from the input data because they are neither targets nor features
Target - Whether a organization is successful in its venture - column "IS_SUCCESSFUL"
Features - All other columnsafter Data Clean up, excluding Target

Compiling, Training, and Evaluating the Model

Started with two layers, then added a third layer in an attempt to improve accuracy.
Made mulitple attempts where i tried changing the number of neurons and activation models.
Noticed that I was not able to achieve the target 75% accuracy even after several attempts.
So after all the attempts for the first Colab output, used the below -

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(120, input_dim=43, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(80, activation="relu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(20, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

Here is the result:
268/268 - 1s - loss: 0.5550 - accuracy: 0.7339 - 673ms/epoch - 3ms/step
Loss: 0.555009663105011, Accuracy: 0.7338775396347046

Next tried another optimization by attempting to leverage keratuner to decide the number of hidden layers and neurons in hidden layers.
That didn't get us to the target accuracy either.
Taking another approach, added a fourth layer and changed the activation -
n = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(200, input_dim=43, activation="tanh"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(150, activation="tanh"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(50, activation="tanh"))

# Fourth hidden layer
nn.add(tf.keras.layers.Dense(20, activation="tanh"))

# Output layer
nn.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Check the structure of the model
nn.summary()

There was a minor improvement from 73.38% to 73.53% with this approach.
Result-
268/268 - 1s - loss: 0.5490 - accuracy: 0.7354 - 565ms/epoch - 2ms/step
Loss: 0.5490406155586243, Accuracy: 0.7353935837745667

Summary: 
Even after multiple attempts was not able to acheive the target accuracy.
At this point it may be worthwhile inspecting the data by re-adding the dropped columns one at a time and see if that gives us a better results.
Additionally we can further look into refining the Kerastuner approach.