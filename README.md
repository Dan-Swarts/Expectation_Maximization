# What's the big Deal?

Do you have 5,000 pictures on your phone? Do you wish that you could fold, or squish them so that you fit more? well we can use some statistical **magic** to do just that!

# How does it work?

A pixel typically takes up 3 bytes of data: 1 byte for the R,G,and B values which make up the color. But the truth is we don't actually need all that space. Colors are sparse: most of them never get used. That's why we don't actually need 3 bytes of data per pixel. 

# How does it work?

We can achieve this by redefining the color-domain from 3x256 potential values to just k values, where k can be as low as 2. We do this by modeling the image as k normal distributions, where each distribution is modeled in 3-dimensional eucledian space, where the axes are the color channels. each distribution has a mean and variance which we will take into consideration. 

# K means 

We will first aproximate the means of the distributions using the statistical K-means algorithm. 