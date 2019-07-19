import cv2
import skimage
import PIL
def overlay_image(dest_image, src_image, x_offset, y_offset):
  #  print(src_image.shape)
  #  print(dest_image.shape)
    rtn = dest_image.copy()
    y1, y2 = y_offset, y_offset + src_image.shape[0]
    x1, x2 = x_offset, x_offset + src_image.shape[1]

    alpha_s = src_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        rtn[y1:y2, x1:x2, c] = (alpha_s * src_image[:, :, c] +
                                  alpha_l * rtn[y1:y2, x1:x2, c])
    return rtn
def resizeImage(image, dsize):
    return cv2.resize(image,(dsize[1], dsize[0]), interpolation = cv2.INTER_AREA)
    