# User Settings
ERODE_STEP = False                                       # erodes potential small pixels
ALTERNATE_HAND_DETECTION = True                         # own method of detecting left/right hand
OUTPUT_FOLDER = 'D:\\Datengrab\\BA_workspace\\out\\'    # output folder for ROI, log & debug pictures
SKIN_SAMPLE = "db\\skin\\std2.jpg"                      # own skin sample for Mode 1
DEBUG_PICTURES = True                                   # Show steps & visualisation of Algorithm

# rotate the picture
ROTATE = 90                                              # 0 , 90 , 180, 270  (all clockwise)

# Mode/Method for translating Greyscale/Color picture into binary form
MODE = 3
# 1: Skin sample comparison in CbCr Colorspace, time consuming 1-2 min/pic
# 2: YCbCr masking (standard values), fast
# 3: OTSU (Standard), fast

# Advanced Settings (!)

# Otsu Thresholding colors              (Mode 3)
OTSU_LOWER = 0                          # Black
OTSU_HIGHER = 255                       # White
OTSU_SKIN_LOWER = [0, 45, 100]          # Lower skin range HSV
OTSU_SKIN_HIGHER = [100, 255, 200]      # Higher skin range HSV

# YCrCb Thresholding colors             (Mode 2)
YCrCb_SKIN_LOWER = [30, 138, 80]          # Lower skin range YCrCb
YCrCb_SKIN_HIGHER = [255, 180, 127]      # Higher skin range YCrCb

# Shifting
PIXEL_OFFSET = 20
PIXEL_OFFSET_NEG = -PIXEL_OFFSET
VALLEY_GAP_OFFSET = 4

# valley checking
V_ALPHA = 10
V_BETA = 10
V_GAMMA = 5

