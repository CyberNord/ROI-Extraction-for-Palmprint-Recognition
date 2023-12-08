# ROI-Extraction-for-Palmprint-Recognition
Extract ROI from handpalm pictures

This is the public repository for the project done for my Bachelor Thesis at JKU Linz in 2022.

## User Manual

The program consists of several files: `main.py` and `meth.py` should not be altered, as they form the core of the program. The file `settings.py` can be modified. To run the program, a database and an output folder must be specified in `settings.py`. Afterward, the file `main.py` can be executed, for instance, through the console. Note that the required libraries (OpenCV) must be installed.

### Input
As input, a folder containing valid images of palms is expected. The image formats *.png and *.jpg have been successfully tested, but it is likely that this program supports all image formats supported by OpenCV. The hands may be in color or grayscale. To ensure a successful analysis, the hands should be presented with the palm facing up and fingers spread. The surface on which the hands lie should have as good a contrast to the hand as possible.

### Output
The program's results are stored in the folder specified in `settings.py`. All processed images are saved in a folder with a unique ID consisting of date and time. Additionally, a log file and a copy of the parameters under which these results were achieved are created.

### settings.py - File
Here, the user can make all relevant settings.

#### User Settings
- `OUTPUT_FOLDER`: The path of the output folder should be specified. Each run receives its folder, which is stored there with a unique ID consisting of date and time.
- `SKIN_SAMPLE`: This setting is only required when running the program in Mode 1, Skin-Color Thresholding. Here, the absolute path to the skin sample is provided.
- `DATABASE`: The path to the folder containing any number of hand images is specified here.
- `ERODE_STEP`: Enable or disable the erode step. This is recommended for a somewhat uneven background and removes small pixels from the image. True = On, False = Off.
- `ALTERNATE_HAND_DETECTION`: Enable or disable alternative differentiation between the right and left hand. It is recommended to leave this setting as True, as in most cases, it improves detection. True = On, False = Off.
- `DEBUG_PICTURES`: Enable or disable debug outputs. If activated, not only the extracted ROI images but also all intermediate steps are saved. Some of the outputs are visualized with additional computational effort. This setting affects the algorithm's speed. True = On, False = Off.
- `ROTATE`: With this setting, you can adjust how many degrees the image should rotate before processing.
- `MODE`: This setting allows the user to set the mode for extracting the binary hand image.
  - Mode 1: Skin-Color Thresholding - Expects a very good, large, and representative skin sample in image format. The analysis and evaluation of an image can take several seconds, depending on the system.
  - Mode 2: Simple Thresholding - In this mode, the user manually enters upper and lower limits for skin colors in advanced settings.
  - Mode 3: Otsu Algorithm - The Otsu Algorithm is the recommended default setting, as no further changes or inputs are required from the user.

#### Advanced User Settings
These settings should preferably not be changed. Here, fundamental variables such as the values of α, β, γ, can be adjusted.

