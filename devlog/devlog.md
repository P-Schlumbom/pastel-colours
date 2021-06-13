# Pastel colour palette generator development log
## 20210610: progress so far
### data preparation
A dataset of pastel colour palettes was scraped from https://colorhunt.co/ (using the "pastel" tag).  
Each palette was encoded in a size-12 vector, that is 3 RGB values for each of the 4 colours in the scheme.
These are ordered as in the original colorhunt scheme, with the first (main) colour coming first, then the second, etc.  
All 191 colour scheme vectors were assembled into a single 191x12 numpy vector and stored, as **pastel_palettes.npy**.  
At first, the thought was to convert natural colour schemes into pastel colour schemes. To this end, 300 photos were selected 
from my own Google Photos account, and the idea was to extract the colour scheme from each. At first, 
I used a Gaussian Mixture Model (GMM) approach to achieve this; all pixels are arranged in a column 
(i.e. generating a set of 3D datapoints) and GMM was used to find 4 clusters in this 3D space. The average of each of these clusters was taken as a colour in the colour scheme.  
This seemed to work alright, although it was compute intensive for larger images.  
Doing a bit of research into colour quantisation (and looking at what colormind http://colormind.io/ does), 
I found the Modified Mean Colour Quantisation (MMCQ) algorithm, which I also implemented. It is notably faster than the GMM approach, and works fine.  
Using MMCQ, each of my 300 images were processed to produce a size-12 vector to represent the colour scheme, and these were assembled and stored much like the pastel schemes into a 300x12 vector.
Additionally, a 300x12 matrix of randomly generated values was also stored as an alternative dataset.  

### GAN training
A GAN was set up to try and generate pastel colour palettes from input palettes. The hope was to have something that could be given an input colour palette and would then convert it into something that fits the distribution of pastel colour palettes.
Since the intention was to *modify* an existing palette to become a pastel palette, the generator was built as a simple resnet: the output of the 2-layer network is added to the original input to produce the final output. 
Thus the network's task is not to just create a pastel palette, but to take an input palette and come up with modifications that will convert it into a pastel colour palette.  
The natural colourscheme dataset turned out not to be helpful, since most photos ended up having a fairly similar scheme 
of blacks, browns, dark greens and the occasional blues. So the GAN just turned them all into the same colour scheme and called it a day. Who'da thunk?  
So the random colour palettes were used instead. Without hyperparameter tuning this produced a generator that converted most - but not all - colours into fairly pastel-shaded versions. However it would also occasionally output decidedly non-pastel colours, 
such as particularly dark, saturated or neon shades. After extensive hyperparameter tuning (which mostly involved reducing the layer size and increasing the batch size to the maximum), eventually a configuration was found which turned out alright results.   
Basically, it just softened every input colour. This is fairly ok, but I was hoping for a network that would also consider the colours holistically and change some to produce a better overall scheme.  
The generator and discriminator always found an equilibrium at a BCE loss of around 0.7.

### thoughts for improvement
There are several ideas I have for improving results (and also stuff that I still want to do).
- **generate randomised inputs:** currently a pre-generated set of randomised colour schemes is being used. Obviously this leaves the network open to overfitting by memorising the inputs (there are only 300). The simple fix for this is to generate the random input during training, like you're supposed to do for GANs anyway. This may or may not make a significant difference, but at any rate would be easy to try.
- **train the GAN to fill in missing colours:** this would be a direct imitation of what they do at http://colormind.io/, but it would certainly be interesting to see if I can get similar results. It might also be worth scraping ALL colour schemes from https://colorhunt.co/, rather than just using the pastel palettes. One potential advantage my approach might have is that it's generating colour RGB values directly instead of creating whole images of colour schemes, which seems to be what colour mind is doing. This should at the very least allow me to train a far smaller network to achieve the same results.
- **convert the RGB values to HSV:** pastel colours are far more clearly defined in HSV space, where they are defined simply by low saturation and high value. This improved representation should dramatically simplify the problem for the network and yield a simpler, more effective solution overall - for all methods proposed here.
- **ignore neural networks entirely and build a conditional statistical model to represent colour schemes:** I've already drawn ot the schematic for how this model would work. Potentially far simpler than a neural network, while at the same time providing far more control. This would involve having a primary colour, which, once chosen, determines the distribution of the secondary colour, which once chosen determines together with the primary colour the distribution of the tertiary colour, etc.

### generate randomised inputs:
Done. No real changes in terms of loss performance, but results seemed worse.

### converting to HSV
- note on HSV conversions: for whatever reason, some websites seem to like to represent hue as an angle, on a scale of 0-360 (while the other two are represented as percentages). skimage however seems to (sensibly) but all values on a 0-1 scale. This is important to bear in mind.
- now training with HSV data...
- doesn't really seem to have improved things much. Might have to look over the code again later...


### modelling the distribution of the palette colours in HSV space
- in **analysis/pastel_palette_modelling.py**
- for now, have built a three-state GMM model for each colour. It seems that this could actually be a very similar problem 
to phoneme recognition, i.e. modelled the same way by a hidden markov model. Might be worth looking at my HMM ASR code for reference.
  - 