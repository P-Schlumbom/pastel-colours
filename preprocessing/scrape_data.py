import re
import numpy as np
from bs4 import BeautifulSoup

#
# Open html file
#

with open("../data/pastel_colour_schemes.html", 'r') as f:
    data = f.readlines()
html_text = ""
for line in data:
    html_text += line.strip()

soup = BeautifulSoup(html_text, 'html.parser')

#
# Collect the data
#
places = ["place c1", "place c2", "place c3", "place c4"]  # the four colour class names
palettes = []  # this will store one palette per element
palette_divs = soup.findAll("div", {"class": "palette"})  # all palette divisions in the html file
colour_re = re.compile(r"\d+,\s\d+,\s\d+")  # regex to extract the colour RGB from the palette class

for i, palette in enumerate(palette_divs):
    palette_list = []
    for colour in places:
        target_string = palette.find("div", {"class": colour})["style"]
        m = re.search(colour_re, target_string)
        palette_list += [int(val) for val in m.group(0).split(",")]
    palettes.append(palette_list)

palettes = np.asarray(palettes)
print(palettes.shape)
print(palettes)

np.save("../pastel_palettes.npy", palettes)

