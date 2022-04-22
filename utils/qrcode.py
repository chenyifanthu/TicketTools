import os
from time import time
import cv2

def get_qrcode_detector(detector_prototxt_path, 
                        detector_caffe_model_path,
                        super_resolution_prototxt_path,
                        super_resolution_caffe_model_path):

    return cv2.wechat_qrcode_WeChatQRCode(detector_prototxt_path, 
                                          detector_caffe_model_path,
                                          super_resolution_prototxt_path,
                                          super_resolution_caffe_model_path)


def recognize_qrcode_opencv(detector, img, 
                            scale_range=(1, 5)):

    """QR code detection and recognition based on OpenCV

    Args:
        detector (cv2.wechat_qrcode_WeChatQRCode): wechat QRCode Detector
        img (ndarray): Image array
        scale_range (tuple, optional): Find the best image scale during 
            recognition process. Defaults to (1, 5).

    Returns:
        results (tuple [str]): The content of all QR codes in the image.
        points (tuple [ndarray]): Corner coordinates of all QR codes in the image.
    """

    for scale in range(*scale_range):
        img_scale = cv2.resize(img, None, fx=scale, fy=scale)
        results, points = detector.detectAndDecode(img_scale)
        if len(results):
            return results, points
    return None, None


if __name__ == '__main__':

    detector = get_qrcode_detector("QRModel/detect.prototxt", "QRModel/detect.caffemodel", 
                                   "QRModel/sr.prototxt", "QRModel/sr.caffemodel")

    data_dir = "/Users/chenyifan/data/发票/qr-test/qrcode/"
    filelist = sorted(os.listdir(data_dir))
    for filename in filelist:
        filename = data_dir + filename
        img = cv2.imread(filename)
        start = time()
        res, points = recognize_qrcode_opencv(detector, img)
        end = time()
        print("File: %s | Content: %s | Elapse: %.2f sec" % (filename, res, end-start))