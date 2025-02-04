import os
import customtkinter as ctk
from tkinter import messagebox
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from color_utils import *
from image_utils import *
from parse_utils import *
from stereo_utils import * 


def main():
    global image_range_var, color_mode_var, shift_var, feature_var, root

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("Anaglyph Image Processor")
    root.geometry("600x400")

    image_range_var = ctk.StringVar(value="1")
    color_mode_var = ctk.StringVar(value="color")
    shift_var = ctk.BooleanVar(value=False)
    feature_var = ctk.StringVar(value="SIFT")

    os.makedirs('result', exist_ok=True)

    pad = 20

    ctk.CTkLabel(root, text="Image Range:").grid(row=0, column=0, sticky="w", padx=pad, pady=10)
    ctk.CTkEntry(root, textvariable=image_range_var).grid(row=0, column=1, padx=pad, pady=10)

    ctk.CTkLabel(root, text="Color Mode:").grid(row=1, column=0, sticky="w", padx=pad, pady=10)
    ctk.CTkComboBox(root, variable=color_mode_var, values=["color", "gray", "half_mix", "red_blue_monochrome", "red_cyan_monochrome", "half-color", "optimized", "dubois"]).grid(row=1, column=1, padx=pad, pady=10)

    ctk.CTkLabel(root, text="Feature detection:").grid(row=1, column=2, sticky="w", padx=10, pady=10)
    ctk.CTkComboBox(root, variable=feature_var, values=["ORB", "SIFT", "FAST", "AKAZE", "All"]).grid(row=1, column=3, padx=pad, pady=10)

    ctk.CTkLabel(root, text="Apply Shift:").grid(row=3, column=0, sticky="w", padx=pad, pady=10)
    ctk.CTkCheckBox(root, variable=shift_var, text="").grid(row=3, column=1, sticky="w", padx=pad, pady=10)

    ctk.CTkButton(root, text="Process", command=process).grid(row=4, column=0, columnspan=2, pady=20)

    root.mainloop()


def process():
    if feature_var.get() == "All":
        process_images("AKAZE")
        process_images("ORB")
        process_images("SIFT")
        process_images("FAST")
        return
    
    process_images(feature_var.get())



def process_images(feature_detection):
    global image_range_var, color_mode_var, shift_var, feature_var, root

    image_range = image_range_var.get()
    color_mode = color_mode_var.get()
    shift = shift_var.get()

    try:
        image_range_parsed = parse_range(image_range)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid image range: {e}")
        return


    for photo_number in image_range_parsed:
        sub_folder = os.path.join('result', f"img_{photo_number}")
        os.makedirs(sub_folder, exist_ok=True)

        left_image_path = os.path.join("L", f"{photo_number}.jpg")
        right_image_path = os.path.join("R", f"{photo_number}.jpg")

        img1, img2 = load_images(left_image_path, right_image_path)

        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        h, w = img1.shape[:2]

        if feature_detection == "SIFT":
            fd = cv.SIFT_create()
            keypoints1, descriptors1 = fd.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = fd.detectAndCompute(gray2, None)

            matches = flann(descriptors1, descriptors2)
            pts1, pts2, good_matches = filter_matches(matches, keypoints1, keypoints2)

        elif feature_detection == "ORB":
            fd = cv.ORB_create(nfeatures=35000)
            keypoints1, descriptors1 = fd.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = fd.detectAndCompute(gray2, None)

            matches = bf_orb(descriptors1, descriptors2)
            pts1, pts2, good_matches = filter_matches_orb(matches, keypoints1, keypoints2)

        elif feature_detection == "FAST":
            fast = cv.FastFeatureDetector_create()
            fast.setThreshold(35) 
            fast.setType(cv.FAST_FEATURE_DETECTOR_TYPE_9_16)

            keypoints1 = fast.detect(gray1, None)
            keypoints2 = fast.detect(gray2, None)

            brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
            keypoints1, descriptors1 = brief.compute(gray1, keypoints1)
            keypoints2, descriptors2 = brief.compute(gray2, keypoints2)

            matches = bf_orb(descriptors1, descriptors2)
            pts1, pts2, good_matches = filter_matches_orb(matches, keypoints1, keypoints2, max_distance=30)


        elif feature_detection == "AKAZE":
            akaze = cv.AKAZE_create(
                threshold=0.0001,      
                nOctaves=10,           
                nOctaveLayers=10       
            )

            keypoints1, descriptors1 = akaze.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = akaze.detectAndCompute(gray2, None)

            matches = bf_orb(descriptors1, descriptors2)
            pts1, pts2, good_matches = filter_matches_orb(matches, keypoints1, keypoints2)

            
        print(f"Using {feature_detection} number matches: ", len(matches))
        print(f"Using {feature_detection} number good matches: ", len(good_matches))
            

        fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        img1_rectified, img2_rectified, H1, H2 = rectify(img1, img2, pts1, pts2, fundamental_matrix)

        max_var, min_var = calculateMaxDif(pts1, pts2, H1, H2, h, w)

        if shift:
            img1_rectified = shiftImage(img1_rectified, max_var, min_var)
            result_path = os.path.join(sub_folder, f"{feature_detection}_{photo_number}_with_shift_{color_mode}.png")
        else:
            result_path = os.path.join(sub_folder, f"{feature_detection}_{photo_number}_no_shift_{color_mode}.png")

        img1_rectified, img2_rectified = crop(img1_rectified, img2_rectified, H1, H2)

        result = createAnaglyph(img1_rectified, img2_rectified, color_mode)
        image = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        cv.imwrite(result_path, image)




if __name__ == "__main__":
    main()