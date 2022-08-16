# Otsu Thresholding colors
OTSU_LOWER = 0                          # Black
OTSU_HIGHER = 255                       # White
OTSU_SKIN_LOWER = [0, 45, 100]          # Lower skin range HSV
OTSU_SKIN_HIGHER = [100, 255, 255]      # Higher skin range HSV

YCrCb_SKIN_LOWER = [30, 138, 80]          # Lower skin range YCrCb
YCrCb_SKIN_HIGHER = [255, 180, 127]      # Higher skin range YCrCb


# Shifting
PIXEL_OFFSET = 20
PIXEL_OFFSET_NEG = -PIXEL_OFFSET
VALLEY_GAP_OFFSET = 4

# Axis
A_HORIZONTAL = 1
A_VERTICAL = 0

# Matrix values
M_VISIBLE = 255
M_CALCULATION = 1

# valley checking
V_ALPHA = 10
V_BETA = 5
V_GAMMA = 5
