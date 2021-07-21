![header](devlog/figs/test_demo_palettes.png)

# Pastel Colours project
## Investigating the generation of pastel colour palettes based on data
This project is inspired by http://colormind.io/, which trained a GAN (Generative Adversarial Network) to generate 
colour palettes. 
I initially began this project because I believed there was some room for improvement in the methodology colormind 
describes, particularly regarding colour representations. I then extended the project further to focus on generating 
pastel colour palettes, because I am a massive sucker for pastel colours. This involved overcoming some interesting 
challenges, due to the relative low number of pastel colour palettes available as data.

## Contents
*   Running Demo
*   Development
    *   Colormind Methodology
    *   Data
    *   GAN Solution
    *   Colour Analysis
    *   Gaussian State Machine Solution

## Running Demo
**train_gaussian_state_model.py** currently loads the model and data and displays some results. The model can either 
train from scratch or load a previously trained model given a path. Currently this must be done by setting the 
appropriate variables at the top of the file. By default, the constant **LOAD_MODEL=False** ensures a new model is 
trained and saved everytime the file is run.

There are several different functions which display different results.  
**display_generations(source_palette)** takes a single source palette and applies 12 different masks to it, generating 
a new palette from the masked source for each mask. All generated palettes are displayed. alognside the original 
and mask, in a grid in a single figure.  
**display_samples()** takes the first 12 palettes in the dataset, masks all but the first (primary) colour, 
and generates a new palette from that. All 12 generations are displayed alongside the original palette in a figure. 
**display_sample()** takes a single palette, masks all but the first (primary) colour, generates a new palette based 
on that primary colour, and displays all three palettes.  
Setting **SAVE_RESULTS** to True will save these figures.

## Development

### Colormind Methodology
Colormind trains a GAN to generate new colour schemes based on a dataset of colour schemes, apparently available via 
adobe. Interestingly, the site's method description seems to suggest that the GAN was trained on *images* of colour 
schemes - that is NxM image arrays where each colour would be represented by a NxN block of pixels with identical RGB 
values. This seems unecessarily wasteful, since with this setup a large proportion of the neural network's capacity 
will always be wasted on learning that all pixels in each colour block have the same value.  
In other words a large number of parameters in the network will always be spent on learning a pattern we already know, 
and as a general rule the more parameters a neural network has the more susceptible it becomes to overfitting. Thus 
by encoding the knowledge we already have - that all the pixels in each colour block will have the same value - into 
the data representation, we can dramatically reduce the number of parameters in the network without eliminating any 
of the patterns we are actually interested in from the dataset. This should help combat overfitting and, if nothing 
else, at least make the network dramatically more efficient for free.  
So in this project I represented each colour palette as a single vector of shape 1xP, where P = 3K and K is the number 
of colours in the palette. That is, each colour is represented by a single RGB vector, and all RGB vectors of the 
palette are then concatenated into a single vector.

### Data

### GAN Solution

### Colour Analysis

### Gaussian State Machine Solution
