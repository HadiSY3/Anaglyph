import cv2 as cv
import numpy as np


def bruteforce_match(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)                              
    return matches

def flann(des1, des2):
    FLANN_INDEX_KDTREE = 1                                                      # spezifiziert dass der "KD Tree algorithmus zu benutzen ist"
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)                  # "trees"  anzahl der Bäume, mehr = aufwendiger aber effektiver
    search_params = dict(checks=50)                                             # "checks" wie oft soll ein Bäum geprüft werden
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)                                   # "k" uns interessieren für jeden deskriptor im bild1 die nächsten k deskriptoren im bild2
    return matches

def filter_matches(matches, keypoints1, keypoints2):
    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Apply Lowe's ratio test
            good.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)  # Query image points
            pts2.append(keypoints2[m.trainIdx].pt)  # Train image points

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, good


def flann_orb(des1, des2):
    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH for binary descriptors
                        table_number=6,  # Number of hash tables
                        key_size=12,  # Size of the hash table
                        multi_probe_level=1)  # Number of probes

    search_params = dict(checks=50)  # Number of checks for the approximate search
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    return matches

def bf_orb(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches

def filter_matches_orb(matches, keypoints1, keypoints2, max_distance=40):
    # Sort the matches based on their distance (ascending order)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = []
    pts1 = []
    pts2 = []

    # Filter matches based on distance threshold (if provided)
    for m in matches:
        if max_distance is None or m.distance < max_distance:
            good_matches.append(m)
            pts1.append(keypoints1[m.queryIdx].pt)  # Get the corresponding keypoint in img1
            pts2.append(keypoints2[m.trainIdx].pt)  # Get the corresponding keypoint in img2

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2, good_matches


def cut(data, percent):
    data = np.array(data)

    lower_limit = np.percentile(data, percent)
    upper_limit = np.percentile(data, 100 - percent)
    
    data = data[(data >= lower_limit) & (data <= upper_limit)]

    return data


def rectify(img1, img2, pts1, pts2, fundamental_matrix):
    h1, w1 = len(img1), len(img1[0])
    h2, w2 = len(img2), len(img2[0])
    _, H1, H2 = cv.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    return cv.warpPerspective(img1, H1, (w1, h1)), cv.warpPerspective(img2, H2, (w2, h2)), H1, H2


def calculateMaxDif(pts1, pts2, H1, H2, h, w):
    #versatz_mitte = []
    versatz_rand = []

    for i in range(len(pts1)):
        point1 = np.squeeze(cv.perspectiveTransform(np.float32(pts1[i]).reshape(1, 1, 2), H1))
        point2 = np.squeeze(cv.perspectiveTransform(np.float32(pts2[i]).reshape(1, 1, 2), H2))

        if point1[0] < w/10 or point1[0] > (w - w/10) or point1[1] < h/10 or point1[1] > (h - h/10):
            versatz_rand.append(point1[0] - point2[0])
        #else:
        #    versatz_mitte.append(point1[0] - point2[0])

    versatz_rand = cut(versatz_rand, 4)
    
    max_rand = np.max(versatz_rand)
    min_rand = np.min(versatz_rand)

    #rot links  -> -
    #rot rechts -> +

    return max_rand, min_rand