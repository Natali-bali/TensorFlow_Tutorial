import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf





#Decision Boundary Plot for multy/binary classification
def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))

  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)

  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())


#Confusion Matrix Plot

"""The following confusion matrix code is a remix of Scikit-Learn's
plot_confusion_matrix function -
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
and Made with ML's introductory notebook -
https://github.com/madewithml/basics/blob/master/notebooks/09_Multilayer_Perceptrons/09_TF_Multilayer_Perceptrons.ipynb

y_pred and y_test should be formated as a vector

"""
def plot_confusion_matrix(y_pred, y_test, figsize = (10, 10), classes = False, text_size = 10):
  # Create the confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0]

  # Let's prettify it
  fig, ax = plt.subplots(figsize=figsize)
  # Create a matrix plot
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)

  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels)

  # Set x-axis labels to bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Adjust label size
  ax.xaxis.label.set_size(20)
  ax.yaxis.label.set_size(20)
  ax.title.set_size(20)

  # Set threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)


# Find the best Learning Rate
"""To figure out the ideal value of the learning rate (at least the ideal value to begin training our model),
the rule of thumb is to take the learning rate value where the loss is still decreasing but not quite flattened
out (usually about 10x smaller than the bottom of the curve).

Model should be compiled with Adam or SGD optimizer
"""
def find_best_learning_rate(model, X_train, y_train, epochs = 100):

    # Create a learning rate scheduler callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch

    # Fit the model (passing the lr_scheduler callback)
    history = model.fit(X_train,
                          y_train,
                          epochs=100,
                          callbacks=[lr_scheduler])

    # Plot the learning rate versus the loss
    lrs = 1e-4 * (10 ** (np.arange(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs. loss")

# Plot Loss and Accuracy for Classification Model
def plot_loss_curves(history):
    """
    Args: TensorFlow model history object
    Returns: separate loss curves for training and validation metrics.
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    #Plot Loss
    plt.plot(epochs, train_loss, label = 'Training Loss')
    plt.plot(epochs, val_loss, label = 'Validation Loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.legend()

    #Plot Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracy, label = 'Training Accuracy')
    plt.plot(epochs, val_accuracy, label = 'Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.legend()

#Import classes names from directories
#Directories should be named as classes
import pathlib
def classes_names(path):
    data_dir = pathlib.Path(path)
    classes_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    print(classes_names)
    return classes_names

#Plot random pictures from every class dirrectory
import random
from keras import preprocessing
def plot_pictures(path, rows_num = 3):
    dir_names = listdir(path)
    plt.figure(figsize=(10,10))
    for dir_name, num in zip(dir_names, range(len(dir_names)-1)):
        files_names = listdir(path + '/' + dir_name)
        random_img = preprocessing.image.load_img(path + '/'+dir_name + '/' + random.choice(files_names))
        plt.subplot(rows_num, int(len(dir_names))/rows_num, num+1)
        plt.imshow(random_img)
        plt.title(dir_name)
        plt.axis('off')

#Create TensorBoard callback function
import datetime
def create_tf_board_callback(dir_name, experiment_name):
  log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  tensorflow_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'Saving TensorBoard log files to {log_dir}')
  return tensorflow_callback

# #Create checkpoint callback function 
# def create_checkpoint(model_num):
#   """Returns checkpoint callback function, args: model number"""
#   checkpoint_filepath = 'checkpoints/checkpoint_'+str(model_num)+'.ckpt'
#   model_checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True,
#                                                           save_weights_only=True, save_freq='epoch',
#                                                           options=None)
#   return model_checkpoints




#Walk through directories and inspect folders
from os import walk, listdir
def walk_through_dir(path):
    for dirpath, dirnames, filenames in walk(path):
        print(f'There are {len(dirnames)} directories, {len(filenames)} files in {dirpath}')


#Unzip file
"""Unzip file
   Args: path to the file"""
import zipfile
def unzip_file(path):
    zip_ref = zipfile.ZipFile(path)
    zip_ref.extractall()
    zip_ref.close()

def create_model(url, num_classes = 10, input_shape = (224, 224, 3)):
  """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.

  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
  """
  feature_extractor_layer = hub.KerasLayer(url,
                                           trainable=False, # freeze the underlying patterns
                                           name='feature_extraction_layer',
                                           input_shape = input_shape)
  #Note if i add layer to model directly, it doent accept input_shape, so first need to create layer, then add it to model

  model = tf.keras.Sequential([feature_extractor_layer,
                              Dense(num_classes,
                                    activation='softmax',
                                    name='output_layer')
                              ])

  return model

#Load and preprocess image
"""Args: path = path to image, img_size = (224,224)"""
def load_and_prep_image(path, img_size = (224,224)):
  img = preprocessing.image.load_img(path)
  plt.imshow(img)
  plt.axis(False)
  img = preprocessing.image.img_to_array(img)
  img = preprocessing.image.smart_resize(img,img_size)
  img = img/255.0
  img = tf.expand_dims(img, axis = 0)
  return img

#Load and predict one image
def load_and_pred(model, path, class_names = classes_names, image_size = (224,224)):
  img = load_and_prep_image(img_size, path)
  pred = model.predict(img)
  print(f'Prediction: {class_names[pred[0].argmax()] }')

def compare_historys(original_history, new_history, initial_epochs=5):
  """
  Compares two model history objects.
  """
  # Get original history measurements
  acc = original_history.history["accuracy"]
  loss = original_history.history["loss"]

  print(len(acc))

  val_acc = original_history.history["val_accuracy"]
  val_loss = original_history.history["val_loss"]

  # Combine original history with new history
  total_acc = acc + new_history.history["accuracy"]
  total_loss = loss + new_history.history["loss"]

  total_val_acc = val_acc + new_history.history["val_accuracy"]
  total_val_loss = val_loss + new_history.history["val_loss"]

  print(len(total_acc))
  print(total_acc)

  # Make plots
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label='Training Accuracy')
  plt.plot(total_val_acc, label='Validation Accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label='Training Loss')
  plt.plot(total_val_loss, label='Validation Loss')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()
