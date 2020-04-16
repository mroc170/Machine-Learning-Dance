
# Everybody Machine Learning Dance Now!

## Project Abstract
Bringing together the disciplines of computer science and dance, our project is focused on dance sequence generation for the human body based off any provided song from the Spotify streaming platform. It utilizes machine learning via neural networks, hours of dance footage, and the spotify API to bring in music data. We aim to inspire new and exciting choreography and at the same time produce a visualization that is in and of itself a piece of art.

## Environment Setup

### Recording Dance via Kinect

All files necessary for recording dance can be found in the **data collection** folder of this repository.
1.  Kinect Sensor v2 hardware (including USB 2.0 connection cord)
2.  [TouchDesigner](https://derivative.ca/download): Additional [information](https://derivative.ca/UserGuide/Kinect) about Kinect 2 Support in TouchDesigner.
3. [Pure Data](https://puredata.info/): The necessary files to open while recording dance are as follows: pdtestDr.Snow.pd, routeskeleton.pd, and toFile.pd. Note that in all of these files, the Pure Data toggles will need to be activated to allow data through the pipeline. 

### Training Data

<ol>
  <li> <a href="https://www.python.org/downloads/" >Python 3.7</a>
    <ul> 
      <li> Download Python and follow the installer instructions </li>
      <li> On windows, make sure to set the PATH globally </li>
      <li> It is recommended to install PIP as well to simplify the installation of the following packages </li>
      <li> Note: it is important to install Python 3.7.x, as PyTorch does not currently support Python 3.8+</li>
    </ul>
  </li>
  <li> <a href="https://jupyter.org/install" >Jupyter Notebooks</a> 
    <ul>
      <li> Follow the above link for instructions on how to setup Notebooks </li>
      <li> Run an instance on your machine to run the project files cell-by-cell </li>
    </ul>
  </li>
  <li> <a href="https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html">pandas</a> 
    <ul>
      <li><code>pip install pandas</code></li>
    </ul>
  </li>
  <li> <a href="https://matplotlib.org/users/installing.html">matplotlib</a>
    <ul>
      <li><code>python -m pip install -U pip</code></li>
      <li><code>python -m pip install -U matplotlib</code></li>
    </ul>
  </li>
  <li> <a href="https://scikit-learn.org/stable/install.html">scikit-learn</a> 
    <ul>
      <li><code>pip install -U scikit-learn</code></li>
    </ul>
  </li>
  <li> <a href="https://spotipy.readthedocs.io/en/2.11.1/#installation">Spotipy</a> 
    <ul>
      <li><code>pip install spotipy --upgrade</code></li>
    </ul>
  </li>
  <li> <a href="https://pytorch.org/get-started/locally/">PyTorch</a>
    <ul><li>Follow the instructions in the link to determine the correct install command</li></ul>
  </li>
</ol>

### Producing Animations

<ol>
  <li> Processing </li>
</ol>

## Other Materials
[Poster](https://drive.google.com/file/d/1I_vOGQIQij2UzXmbQAJTZQCqG5-yDhEb/view?usp=sharing)

[Drive Folder with Milestones and Progress Presentations](https://drive.google.com/drive/folders/1WIof7IkIQthz4JQYmHtaGFFs6BY_jjKF?usp=sharing) 
