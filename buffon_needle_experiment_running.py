import argparse
import logging
from pathlib import Path
import paho.mqtt.publish as publish

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import signal
from skimage import morphology
from skimage.feature import canny # noqa
from skimage.transform import probabilistic_hough_line

matplotlib.use('tkagg')
logger = logging.getLogger(__name__)


class PointMarker:
    def __init__(self, image: np.ndarray):
        self.img = image
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.points = []
        self.point_objects = []
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.axis('off')
        plt.title('Click to mark 4 points')
        plt.show()

    def onclick(self, event):
        # Only consider clicks inside the axes
        if event.inaxes != self.ax:
            return

        # Create/mark point
        if event.button == 1 and event.key == 'shift':
            if len(self.points) < 4:
                x, y = event.xdata, event.ydata
                self.points.append([x, y])
                pt_obj, = self.ax.plot(x, y, 'bo')  # red dot
                self.point_objects.append(pt_obj)

            if len(self.points) == 4:
                plt.title('4 points marked')

        # Delete last point
        if event.button == 3:
            if len(self.points) > 0:
                self.point_objects[-1].remove()
                del self.point_objects[-1]
                del self.points[-1]
                self.fig.canvas.draw()

            if len(self.points) < 4:
                plt.title('Click to mark 4 points')

        # Update plot
        self.fig.canvas.draw()


def extract_stripe_bounds(image: np.ndarray, show_plots=False):
    if len(image.shape) == 3:
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_bw = image.copy()

    image_margined = np.sum(image_bw, axis=0).astype(float)

    x_corrected = image_margined - np.mean(image_margined)
    ac = np.correlate(x_corrected, x_corrected, mode='full')[len(x_corrected) - 1:]
    peaks, _ = signal.find_peaks(ac)
    stripe_length = peaks[0] // 2
    logger.log(logging.INFO, f"stripe_length: {stripe_length}")

    if show_plots:
        plt.figure()
        plt.plot(image_margined)
        plt.title(f"Margined image, Estimated stripe length: {stripe_length} pxs.")

    return stripe_length


def rectify_needles(needles, ideal_length):
    needles = np.array(needles, dtype=np.float32)

    needles_length = np.sqrt(
        np.sum(np.diff(needles, axis=1) ** 2, axis=2)
    )

    needles_sum = np.sum(needles, axis=1, keepdims=True)
    needles_centered = needles - (needles_sum / 2) # needles translated to their center based coordinate system
    mul_fac = ideal_length / needles_length
    needles_centered *= mul_fac[:, np.newaxis] # needles are corrected

    return needles_centered + (needles_sum / 2)


def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="Buffon's experiment - WHAT DO THE TOOTHPICKS SAY?.")
    parser.add_argument('--hide_misc_plots', action='store_true', help="Raise this flag to hide plots")
    parser.add_argument('-log', '--loglevel', default='info', help='Provide logging level')

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel.upper())
    show_plots = not args.hide_misc_plots

    try:
        logger.log(logging.INFO, parser.description)

        while True:
            input_path = input("Image path: ")
            if not Path(input_path).is_file():
                logger.log(logging.WARNING, f"Input path {input_path} doesn't exist")
                continue

            image = cv2.imread(input_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ## Image rectification
            marker = PointMarker(image)

            crop_points = marker.points
            if len(crop_points) < 4:
                logger.log(logging.ERROR, f"Cropped points: {crop_points}")
                continue
            logger.log(logging.INFO, f"Crop points: {crop_points}")

            new_w, new_h = (960, 762)  ### PARAM

            new_points = [[0, 0],
                          [new_w, 0],
                          [0, new_h],
                          [new_w, new_h]]

            aff_trans = cv2.getPerspectiveTransform(np.float32(crop_points)[:4], np.float32(new_points)[:4])
            rectified_image = cv2.warpPerspective(image, aff_trans, (new_w, new_h))
            logger.log(logging.INFO, f"Rectified image shape: {rectified_image.shape}")

            resize_factor = 1  ### PARAM
            new_shape = np.array(rectified_image.shape)[:2][::-1] // resize_factor
            rectified_image = cv2.resize(rectified_image, tuple(new_shape))
            logger.log(logging.INFO, f"Resized image shape: {rectified_image.shape}")

            ## Calculating stripe length in pixels
            stripe_length = 64 #extract_stripe_bounds(rectified_image, show_plots=show_plots)  ### PARAM

            ## Image preprocessing
            # Sharpening edges
            rectified_image_blur = cv2.GaussianBlur(rectified_image, ksize=(21, 21), sigmaX=1)
            strength = 0.5
            rectified_image = cv2.addWeighted(rectified_image, 1.0 + strength, rectified_image_blur, -strength, 0)

            # Binarization
            rectified_image_std = np.std(rectified_image, axis=2)
            rectified_image_std = cv2.normalize(rectified_image_std, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)
            bin_thresh = np.quantile(rectified_image_std.flatten(), 0.975)  ### PARAM
            rectified_image_bin = cv2.threshold(rectified_image_std, float(bin_thresh), 255.0, cv2.THRESH_BINARY)[1]
            if show_plots:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
                ax = axes.ravel()

                ax[0].imshow(rectified_image_std, cmap=cm.gray)
                ax[0].set_title('Standard deviation')

                ax[1].imshow(rectified_image_bin, cmap=cm.gray)
                ax[1].set_title('Binary image')

            ## Detecting needles
            # edges = canny(rectified_image_bin, 0)
            skeletons = morphology.skeletonize(rectified_image_bin)
            lines = probabilistic_hough_line(skeletons, ### PARAM
                                             threshold=10,
                                             line_length=round(0.5 * stripe_length),
                                             line_gap=3,
                                             rng=3)
            logger.log(logging.INFO, f"Number of detected needles: {len(lines)}")

            if show_plots:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
                ax = axes.ravel()

                ax[0].imshow(rectified_image_bin, cmap=cm.gray)
                ax[0].set_title('Binary image')

                ax[1].imshow(skeletons, cmap=cm.gray)
                ax[1].set_title('Skeletons')

                ax[2].imshow(rectified_image)
                for line in lines:
                    p0, p1 = line
                    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
                ax[2].set_xlim((0, rectified_image_bin.shape[1]))
                ax[2].set_ylim((rectified_image_bin.shape[0], 0))
                ax[2].set_title('Probabilistic Hough')
                plt.show(block=False)

            needles = rectify_needles(lines, ideal_length=stripe_length)

            ## Which needles have crossed ???
            needles_xloc = needles[..., 0] // stripe_length
            needles_crossed = needles_xloc[:, 0] != needles_xloc[:, 1]
            pi_value = (2 * len(needles) ) / np.sum(needles_crossed)

            logger.log(logging.INFO, f"Needles crossed: {np.sum(needles_crossed)}")
            logger.log(logging.INFO, f"Estimate for pi from this image is (DRUM ROLL...): {pi_value}")

            plt.figure()
            plt.imshow(rectified_image)
            plt.title(f"Estimated pi value is: {pi_value}")
            h, w = rectified_image.shape[:2]
            # stripe lines
            for i in range((w // stripe_length) + 2):
                if i * stripe_length <= w:
                    plt.axvline(x=i * stripe_length, color='blue')
            # needles
            for line, n_crossed in zip(needles, needles_crossed):
                p0, p1 = line
                color = 'cyan' if n_crossed else 'green'
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color=color)
            plt.show(block=False)


            save_status = input("Do you want to save the result? (y/n): ")
            if save_status == 'y':
                # Plotting and saving he estimates
                publish.single(
                    "buffon pi",
                    payload=f"{len(needles)}_{np.sum(needles_crossed)}_{pi_value}", 
                    hostname="test.mosquitto.org"
                )

    except KeyboardInterrupt:
        logger.log(logging.ERROR, f"User interrupted.")


if __name__ == "__main__":
    main()
