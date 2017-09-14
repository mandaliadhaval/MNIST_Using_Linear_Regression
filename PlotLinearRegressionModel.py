
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST/", one_hot=True)


# In[3]:


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))


# In[ ]:


data.test.labels[0:5, :]


# In[ ]:


data.train.labels[0:5, :]


# In[ ]:


data.validation.labels[0:15, :]


# In[7]:


data.test.clas = np.array([label.argmax() for label in data.test.labels])


# In[ ]:


data.test.clas[0:10]


# In[4]:


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of classes, one class for each of 10 digits.
num_classes = 10


# In[5]:


def plot_images (images, clas_true, clas_pred=None):
    assert len(images)== len(clas_true)==12
    #Create a 4x3 grid for images
    fig, axes = plt.subplots(4,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        # Plot images
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        
        #Show both true and predicted classes
        if clas_pred is None:
            xlabel = "True : {0}".format(clas_true[i])
        else:
            xlabel = "True : {0}, Pred: {1}".format(clas_true[i], clas_pred[i])
                
        ax.set_xlabel(xlabel)
            
        #Remove Ticks from the Plot
        ax.set_xticks([])
        ax.set_yticks([])


# In[8]:


# Get images from test set
images = data.test.images[0:12]

#Get respective true class of the images above
clas_true = data.test.clas[0:12]

#Plot the images and labels
plot_images(images=images, clas_true=clas_true)


# In[9]:


#img_size_flat = img_size*img_size, img_size=28
x = tf.placeholder(tf.float32, [None, img_size_flat])


# In[10]:


#num_classes = 10
y_true = tf.placeholder(tf.float32, [None, num_classes])


# In[11]:


#reclassify as integer
y_true_clas = tf.placeholder(tf.int64, [None])


# In[12]:


# Identify weights that can be shown as a heat map corresponding to the number. These weights genrate probability of a positive response over a 28X28 pixel size
# This is initalized at zero
weights = tf.Variable(tf.zeros([img_size_flat,num_classes]))


# In[13]:


# Identify biases in the linear regression equation y-xw + 'b'. This is a flat or constant vector or tensor with only one dimension
bias = tf.Variable(tf.zeros([num_classes]))


# In[14]:


# Generate logits with equation y=xw+b
logits = tf.matmul(x,weights)+bias
# Here logits is a matrix of 784 rows and 10 classes. Where the the element of ith row and jth class defines the probability of ith input image is likely close to jth class.


# In[15]:


# These logits need to be normalised using the maximum used function of softmax, almost used in every model as a final layer.
# Here y_pred is predicted out put
y_pred = tf.nn.softmax(logits)


# In[16]:


# To predict the highest probability using predicted class
y_pred_clas = tf.argmax(y_pred, dimension=1)


# In[17]:


# We need to randomize this function in order to train and find the best performance. To evaluate performance we need to use the performance measure known as cross_entropy. Cross_entropy is an OOB function.
# It is always positive indicating randomness. In order to make the model efficient our goal will be to minimize cross_entropy as much as possible
# Here Y_True is our flat placeholder initialized earlier to hold classes as y_true = tf.placeholder(tf.float32, [None, num_classes])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)


# In[18]:


# Now to reduce the cross entropy we take the mean which is also a single scalar value and easier to measure
cost = tf.reduce_mean(cross_entropy)


# In[19]:


# Goal for the model will be reduce cost. To optimize that we carry out a Gradient Descent to find the local minimum. Learning rate can used between 0.1 to 0.5.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)


# In[20]:


# Lets create two more measures so that we can plot the overall progress where predicted class becomes equal to true class
# Remeber we had initialized y_true_clas as a flat integer placeholder y_true_clas = tf.placeholder(tf.int64, [None])
correct_pred=tf.equal(y_pred_clas,y_true_clas)


# In[21]:


# For each of this classification lets calculate the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[22]:


# Till now everything has been more or less placeholder. Lets initialize Tensorflow variables now.
sess=tf.InteractiveSession()


# In[23]:


sess.run(tf.global_variables_initializer())


# In[24]:


# Lets create a knob by which we can control the batch size
batch_size = 100


# In[25]:


# Lets create optimizer function by which model can be trained to change weights and biases to improve the accuracy. In each iteration based on the batch size,
# a new batch of data will be selected from training-set and tensorflow will execute the optimizer function.
# x_batch will hold the images and y_true_batch will hold the true values for those images. Put the batch into dictation to hold names for the placeholder variables in tensorflow graph
# Remember we are using training data set here, so your placeholders need to be accordingly used
def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x : x_batch, y_true: y_true_batch}
        sess.run(optimizer, feed_dict=feed_dict_train)
        
        


# In[26]:


# Lets also initialize the feed for test data
feed_dict_test = {x: data.test.images, y_true: data.test.labels,
                  y_true_clas: data.test.clas}


# In[27]:


# Function for printing the classification accuracy on the test-set.
def print_accuracy():
    acc = sess.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy of test-data: {0:.1%}".format(acc))


# In[28]:


# Plot the accuracy using confusion matrix
def print_confusion_matrix():
    # Identify true classification from test set
    clas_true = data.test.clas

    # Identify the predicted classification from test set
    clas_pred = sess.run(y_pred_clas, feed_dict=feed_dict_test)

    # Plot confusion matrix
    cx = confusion_matrix(y_true=clas_true,
                     y_pred=clas_pred)
    # Print confusion matrix
    print(cx)

    # Plot confusion matrix as image
    plt.imshow(cx, interpolation='nearest', cmap=plt.cm.Greens)

    # Format the layout
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')


# In[48]:


# Plot images that have been misclassified 
# Use TensorFlow to get a list of boolean values
# whether each test-image has been correctly classified,
# and a list for the predicted class of each image.

def plot_example_errors():
    correct, clas_pred = sess.run([correct_pred, y_pred_clas],
                                    feed_dict=feed_dict_test)
    print("Correctly Classified Numbers:")
    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    clas_pred = clas_pred[incorrect]

    # Get the true classes for those images.
    clas_true = data.test.clas[incorrect]
    
    # Plot the first 12 images.
    plot_images(images=images[0:12],
            clas_true=clas_true[0:12],
            clas_pred=clas_pred[0:12])


# In[44]:


# Plot weights of the model for evaluation
def plot_weights():
    w = sess.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i<10:
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel("Weights: {0}".format(i))

            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')
        # Remove ticks from each sub-plot.
        ax.set_xticks([])
        ax.set_yticks([])


# In[49]:


def plot_examples():
    correct, clas_pred = sess.run([correct_pred, y_pred_clas],
                                    feed_dict=feed_dict_test)
    print("Correctly Classified Numbers:")
    # Negate the boolean array.
    incorrect = (correct == True)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    clas_pred = clas_pred[incorrect]

    # Get the true classes for those images.
    clas_true = data.test.clas[incorrect]
    
    # Plot the first 12 images.
    plot_images(images=images[0:12],
            clas_true=clas_true[0:12],
            clas_pred=clas_pred[0:12])


# In[32]:


print_accuracy()


# In[33]:


plot_examples()


# In[34]:


plot_example_errors()


# In[45]:


print_confusion_matrix()
plot_weights()


# In[46]:


optimize(num_iterations=10)


# In[47]:


print_accuracy()
plot_examples()
plot_example_errors()


# In[50]:


plot_weights()


# In[51]:


print_confusion_matrix()


# In[52]:


optimize(num_iterations=10)


# In[53]:


print_accuracy()
plot_examples()
plot_example_errors()


# In[ ]:


plot_weights()



# In[ ]:


print_confusion_matrix()


# In[54]:


optimize(num_iterations=100)


# In[55]:


print_accuracy()
plot_examples()
plot_example_errors()


# In[56]:


plot_weights()


# In[ ]:


print_confusion_matrix()


# In[ ]:


#sess.close()

