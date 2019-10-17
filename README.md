# Convolution
<b>Convolution:</b>
A Convolution is combined integration of two functions and its show that how one functions modified the shape of the others. 
Eg. Signal processing in electrical engineering. 
It is a combination of input image as well as feature detector (Kernel, Filter). 
Please check some basic parameters below which we are using in convolutional layer.
Click <a href="http://mlaileader.com/post/convolution/7">here</a>. to check the details implementation

  <b>•	Size of the input image:</b> ( E.g. n= 7 x 7 x nc) nc Number of channel (RGB colored nc = 3)

  <b>•	Padding (P =1 or 2):</b> Its very useful when we have important feature on edge (it can give us same size image 
  (same padding or half padding or full padding)

  <b>•	Strides (S= 2, 3):</b>  we can use strides when we don’t need to each and every feature (By defaults its one but we may change 
  it to 2 or 3 depend upon feature detector importance)
  
  <b>•	Filter size (fc=3 x 3 x nc or 5 x 5 x nc):</b> filter size again depends upon the image size and we would like to filter the 
  image feature. Generally, we are using 3 x 3 feature detector.

<b>Formula for input image and output image:</b>
[n x n x nc , n x n x nc ] * [fc x fc x nc]  here * is called the convolutional operation it will work as element wise multiplication. 

[ (n x n x nc) ,(n x n x nc)] --> input image size 

[ ((n + 2P – fc)/S) +1, ((n + 2P – fc)/S) +1] --> Output image (Feature map or convolved feature)

We may create the multiple feature maps to obtain our first convolutional layer. 
For creation multiple feature map, we need to do convolutional operation with multiple feature detectors.
