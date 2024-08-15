Code to run a WM match-to-sample task with sequential presentations, with different locations and categories cued on each trial.


To run the script, navigate to this directory in the terminal & run the following lines :

```
conda activate psychopy  # activates the appropriate conda environment with a working psychopy install
python task_oldpy.py
```

*Note*: make sure a `data/` directory also exists in the same directory as the task script; it's where the data will be saved.

Here's a description of the files:

`task.py` was the original script.

`task_oldpy.py` is a modified version that runs on the scanner laptop, in the `psychopy` conda environment. It contains all of the final tweaks and fixes, so it should be the reference for generating new scripts.

`images/` is a nested directory containing the 3 main image categories, and the 8 pairs of images for each category.

`images_scrambled/` follows the same structure as `images/`, and contains the phase scrambled versions of the original images.

`imscramble.m` is a matlab script used to produce the phase scrambled images. Note that it won't run now, as it expects to be pointed to directories with flat structures, rather than nested.
Note, the code was taken from Martin Hebart's website: [http://martin-hebart.de/webpages/code/stimuli.html](http://martin-hebart.de/webpages/code/stimuli.html)