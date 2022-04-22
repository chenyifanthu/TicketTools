import cv2
from yolo.yolov5 import YOLOv5


# 加载模型
detect_model = YOLOv5(weights='/Users/chenyifan/Desktop/yolov5m_ticket_0401.pt', 
                      imgsz=(1088, 1088),
                      device='cpu', conf_thres=0.36)
print("Class Name:", detect_model.model.names)

# 单帧图像检测
img_path = 'test-fp.jpg'
img = cv2.imread(img_path)
bboxs, scores, cls = detect_model.detect(img) # 检测图片
print(cls)
print(bboxs)
plot_img = detect_model.draw_result(img, bboxs, scores, cls) # 绘制结果
cv2.imwrite('output.jpg', plot_img)

# 文件夹内所有图片检测
# import glob, os
# os.makedirs("test_results/", exist_ok=True)
# filelist = glob.glob('/Users/chenyifan/data/表格/表格yolo测试/*.*')
# for file in filelist:
#     img = cv2.imread(file)
#     bboxs, scores, cls = detect_model.detect(img)
#     plot_img = detect_model.draw_result(img, bboxs, scores, cls)
#     cv2.imwrite(os.path.join('test_results', os.path.basename(file)), plot_img)
    
    