# Flow
Flow! Advanced Algorithms Final Project

Tommy Liu and Kimberly Lopez
Harvey Mudd College

In this project, we implement two linear programming algorithms to solve rectangular Flow instances! We model Flow as a Conenctivity problem and as a Multi-Commodity Flow problem. 


Codebase Architecture:

We include both the notebook version of our code and an all-in-one python script which can be run individually.

The architecture of the codebase is the following

Our Project
│
├── flow.ipynb                  # This is the notebook demonstration of our linear programs, which calls functions from flow.py
│
├── flow.py                     # This stores all the functions needed for the notebook flow.ipynb
│
├── main.py                     # This is the all-in-one python script that solves the flow problem and provides visualization
│
├── visualization.png           # This is file output by main.py
│
├── make_images.py              # This is the file that converts the black background to white, which is useful in making handouts.
│
├── examples/                   # This folder is the directory to upload screenshots of flow instances to solve.
│
├── images/                     # This folder is the directory to upload screenshots of flow instances with black background.
│
└── processed_images/           # This folder is the directory to store the output images with their background turned white.



When the user uses the projects (main.py or flow.ipynb), three essential places to change are
1. num_rows - the number of rows of the flow instance
2. num_cols - the number of columns of the flow instance
3. impath - this is the argument for the function load_img(impath), which is the path to the image relative to the folder examples/