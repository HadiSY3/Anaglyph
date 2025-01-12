import numpy as np
import cv2 as cv

def createAnaglyph(img_left, img_right, mode="color"):
    if mode == "color":
        anaglyph = (img_left * np.array([1, 0, 0])) + (img_right * np.array([0, 1, 1]))
    elif mode == "gray":
        left_gray = np.dot(img_left[..., :3], [0.299, 0.587, 0.114])
        right_gray = np.dot(img_right[..., :3], [0.299, 0.587, 0.114])
        anaglyph = np.stack([left_gray, right_gray, right_gray], axis=-1)
    elif mode == "half_mix":
        left_gray = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        right_gray = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
        leftCvt = cv.cvtColor(left_gray, cv.COLOR_GRAY2RGB)
        rightCvt = cv.cvtColor(right_gray, cv.COLOR_GRAY2RGB)
        finalLeft = (0.5 * leftCvt + 0.5 * img_left)
        finalRight = (0.5 * rightCvt + 0.5 * img_right)
    elif mode == "red_blue_monochrome":
        left_gray = np.dot(img_left[..., :3], [0.299, 0.587, 0.114])
        right_gray = np.dot(img_right[..., :3], [0.299, 0.587, 0.114])
        anaglyph = np.stack([left_gray, np.zeros_like(left_gray), right_gray], axis=-1)
    elif mode == "red_cyan_monochrome":
        left_gray = np.dot(img_left[..., :3], [0.299, 0.587, 0.114])
        right_gray = np.dot(img_right[..., :3], [0.299, 0.587, 0.114])
        anaglyph = np.stack([left_gray, right_gray, right_gray], axis=-1)
    elif mode == "half-color":
        anaglyph = halfColorAnaglyphMode(img_left, img_right)
    elif mode == "optimized":
        anaglyph = optimizedAnaglyphMode(img_left, img_right)
    elif mode == "dubois":
        anaglyph = duboisAnaglyphMode(img_left, img_right)
    else:
        raise ValueError(f"Unbekannter Modus: {mode}")

    anaglyph = anaglyph.astype('uint8')
    return anaglyph

def optimizedAnaglyphMode(img_left, img_right):
    # Red channel: 30% green + 70% blue from the left image, brightened by 50%
    red_channel = (img_left[:, :, 1] * 0.3 + img_left[:, :, 2] * 0.7) * 1.5
    red_channel = np.clip(red_channel, 0, 255)  # Ensure values are within valid range (0-255)

    # Green channel: 100% of the green channel from the right image
    green_channel = img_right[:, :, 1]

    # Blue channel: 100% of the blue channel from the right image
    blue_channel = img_right[:, :, 2]

    # Combine channels into the final output image
    output = np.zeros_like(img_left)
    output[:, :, 0] = red_channel.astype(img_left.dtype)  # Assign to red channel
    output[:, :, 1] = green_channel                      # Assign to green channel
    output[:, :, 2] = blue_channel                       # Assign to blue channel

    return output

    
def halfColorAnaglyphMode(img_left, img_right):
    # Grayscale weights for the left image (grayscale representation for red channel)
    grayscale_weights = np.array([0.299, 0.587, 0.114])
    left_grayscale = np.dot(img_left, grayscale_weights)[:, :, np.newaxis]  # Compute grayscale image

    # Green and blue channels from the right image
    right = img_right * np.array([0, 1, 1])

    # Combine left grayscale into red channel and right green and blue channels
    output = np.zeros_like(img_left)
    output[:, :, 0] = left_grayscale[:, :, 0]  # Assign grayscale to red channel
    output[:, :, 1:] = right[:, :, 1:]        # Assign right image's green and blue channels

    return output

def duboisAnaglyphMode(img_left, img_right):
    # Compute the red channel
    red_channel = (
        img_left[:, :, 0] * 0.456 +
        img_left[:, :, 1] * 0.500 +
        img_left[:, :, 2] * 0.176 +
        img_right[:, :, 0] * -0.043 +
        img_right[:, :, 1] * -0.088 +
        img_right[:, :, 2] * -0.002
    )
    
    # Compute the green channel
    green_channel = (
        img_left[:, :, 0] * -0.040 +
        img_left[:, :, 1] * -0.038 +
        img_left[:, :, 2] * -0.016 +
        img_right[:, :, 0] * 0.378 +
        img_right[:, :, 1] * 0.734 +
        img_right[:, :, 2] * -0.018
    )
    
    # Compute the blue channel
    blue_channel = (
        img_left[:, :, 0] * -0.015 +
        img_left[:, :, 1] * -0.021 +
        img_left[:, :, 2] * -0.005 +
        img_right[:, :, 0] * -0.072 +
        img_right[:, :, 1] * -0.113 +
        img_right[:, :, 2] * 1.226
    )
    
    # Combine channels into a final output image
    output = np.zeros_like(img_left, dtype=np.float32)
    output[:, :, 0] = red_channel
    output[:, :, 1] = green_channel
    output[:, :, 2] = blue_channel

    # Clip values to ensure they're within the valid range [0, 255]
    output = np.clip(output, 0, 255).astype(img_left.dtype)

    return output
