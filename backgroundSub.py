import cv2 as cv
import numpy as np
from objTrack import CentroidTracker
import time


def maskMotion(mask, minimum=0, array=np.array((9, 9), dtype=np.uint8)):
    """
        Process a binary mask to enhancce motion areas by using thresholding and morphological operations.

        Function Parameters:
            mask (numpy.ndarray): The input binary mask, a gray frame where the foreground
                                    represents motion (or lack thereof).

            minimum (int): The threshold value to create a binary image from the mask. Default is 0.

            array (numpy.ndarray): The structuring element used for morphological operations.

        Returns:
            numpy.ndarray: The processed mask which enhances regions of motion, with reduced noise.
    """

    # apply threshold to the mask where pixels > 255 are white, everything else is black
    tmp, threshold = cv.threshold(mask, minimum, 255, cv.THRESH_BINARY)

    # apply morphology to remove some noise from the video
    motionMsk = cv.morphologyEx(threshold, cv.MORPH_OPEN, array, iterations=2)
    motionMsk = cv.morphologyEx(motionMsk, cv.MORPH_CLOSE, array, iterations=2)

    return motionMsk


def findContours(frame):
    """
        Find contours in a binary frame and draw bounding boxes around significant contours.

        Function Parameters:
            frame (numpy.ndarray): The binary image to find the contours

        Returns:
            list: A list of bounding boxes where each box is represented by a numpy array giving the four corners
            ex. [x, y, x+w, y+h].
    """

    contours, heir = cv.findContours(frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # Initialize an empty list to store numpy arrays of points
    boxes = []
    # loop through the contours and if they are large enough, bound them with a box
    for cnt in contours:
        if cv.contourArea(cnt) > 1000:
            x, y, w, h = cv.boundingRect(cnt)
            box = np.array(([x, y, x+w, y+h]))
            boxes.append(box.astype(int))

    return boxes


def main():
    """
        Main function to capture video, process frames to detect motion, find contours,
        and track objects. It also calculates the velocity of objects passing between
        two defined lines in the frame.
    """
    # initialize the background subtractor
    backSub = cv.createBackgroundSubtractorKNN(history=1, dist2Threshold=1000, detectShadows=False)

    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Unable to open")
        exit(0)

    # initialize the tracker centoids
    ct = CentroidTracker()

    # uncomment and add your known distance in the frame
    distance = .002

    # make a dict to hold the centroids found and their times
    velCentroids = {}

    while True:

        ret, frame = capture.read()

        # apply background subtraction, get the foreground mask, then find the contours of the frame
        fgMask = backSub.apply(frame)
        motion = maskMotion(fgMask)
        rects = findContours(motion)

        # update the centroid tracker with the new-found bounding boxes
        centoids = ct.update(rects)

        # These are the lines that we use to calculate velocity, change them to fit your known distance in the frame
        # cv.line(frame, (200, 1), (200, 2000), (0, 0, 255), 5)
        # cv.line(frame, (600, 1), (600, 2000), (0, 0, 255), 5)

        # now we loop through each centorid found with the tracking algorithm
        # if the centroid is within the lines and it is not in
        for (objectID, centroid) in centoids.items():
            
            if 200 < centroid[0] < 600:
                # if the centroid is within the set range, start the clock and add the
                # centroid to the velCentroid dict for later processing
                if objectID not in velCentroids:
                    start_time = time.time()
                    velCentroids[objectID] = start_time

            elif objectID in velCentroids:
                # if the centroid left the set range then we calculate its velocity
                # through the set distance given earlier and print the velocity
                if (centroid[0] < 200) or centroid[0] > 600:
                    endtime = time.time()
                    totalTime = endtime - velCentroids[objectID]
                    vel = (distance/totalTime) * 3600
                    print(f"velocity is {vel} mph")
                    del velCentroids[objectID]
                    objectID = vel

            # circle the centroid
            cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        cv.imshow('Frame', frame)

        if cv.waitKey(1) == ord('q'):
            break
    capture.release()


if __name__ == '__main__':
    main()
