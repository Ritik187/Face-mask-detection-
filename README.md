<h1 align="centre"> Face Mask Detection</h1>

### Contents
* [About](#About)
* [Dataset](#Dataset)
* [Libraries](#Libraries-Used).
* [Algorithm used](#Algorithm-used)
* [Notebooks](#Notebooks-and-their-links)
* [Implementation tool](#Implementation_tool)
* [Result](#Result)
* [Author](#Author)

### About
It is a deep learning model and aims to detect whether the person is wearing mask or not.
 * This model has an accuracy about 93%(approx).
 * It sent an alert mail if without mask is detected.

### Dataset
* [Link](https://github.com/Ritik187/Face-mask-detection-/tree/master/data)

### Libraries Used
<ul>
  <li>OpenCV</li>
  <li>Matplotlib</li>
  <li>Numpy</li>
  <li>Pandas</li>
  <li>Sklearn</li>
  <li>Tensorflow</li>
  <li>Smtplib</li>
 </ul>

### Algorithm used.
I have used Convulational Neural Network which is a deep learning algorithm to train the model.
  
### Notebooks and their links
* [mask_detection.py](https://github.com/Ritik187/Face-mask-detection-/blob/master/mask_detection.py)
* [mask_detection.json](https://github.com/Ritik187/Face-mask-detection-/blob/master/mask_detection.json)
* [mask_detection.h5](https://github.com/Ritik187/Face-mask-detection-/blob/master/mask_detection.h5)
* [haarcascade_frontalface_default.xml](https://github.com/Ritik187/Face-mask-detection-/blob/master/haarcascade_frontalface_default.xml)
* [live_detection.py](https://github.com/Ritik187/Face-mask-detection-/blob/master/live_detection.py)


### Implementation tool
I used Jupyter Notebook to implement all these files for this model.

### Result
This model gave 93% accuracy to detect face without mask after training via tensorflow-gpu==2.0.0

### Author
<li>Ritik Meena</li>







