import cv2
import numpy as np
import time

from ocr import utility
from ocr.data import create_operators, transform
from ocr.postprocess import build_post_process



class TableStructurer(object):
    def __init__(self, args):
        pre_process_list = [{
            'ResizeTableImage': {
                'max_len': args.table_max_len
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'PaddingTableImage': None
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image']
            }
        }]
        postprocess_params = {
            'name': 'TableLabelDecode',
            "character_type": args.table_char_type,
            "character_dict_path": args.table_char_dict_path,
        }

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            utility.create_predictor(args, 'table')

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img = data[0]
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        img = img.copy()
        starttime = time.time()

        self.input_tensor.copy_from_cpu(img)
        self.predictor.run()
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)

        preds = {}
        preds['structure_probs'] = outputs[1]
        preds['loc_preds'] = outputs[0]

        post_result = self.postprocess_op(preds)

        structure_str_list = post_result['structure_str_list']
        res_loc = post_result['res_loc']
        imgh, imgw = ori_im.shape[0:2]
        res_loc_final = []
        for rno in range(len(res_loc[0])):
            x0, y0, x1, y1 = res_loc[0][rno]
            left = max(int(imgw * x0), 0)
            top = max(int(imgh * y0), 0)
            right = min(int(imgw * x1), imgw - 1)
            bottom = min(int(imgh * y1), imgh - 1)
            res_loc_final.append([left, top, right, bottom])

        structure_str_list = structure_str_list[0][:-1]
        structure_str_list = ['<html>', '<body>', '<table>'] + structure_str_list + ['</table>', '</body>', '</html>']

        elapse = time.time() - starttime
        return (structure_str_list, res_loc_final), elapse

if __name__ == "__main__":
    args = utility.parse_args()
    
    args.det_model_dir = "./models/ch_ppocr_server_v2.0_det_infer/"
    args.rec_model_dir = "./models/ch_ppocr_server_v2.0_rec_infer/"
    args.table_model_dir = "./models/en_ppocr_mobile_v2.0_table_structure_infer/"
    args.use_gpu = False
    
    table_structurer = TableStructurer(args)

    image_file = 'test-img-2.jpeg'
    img = cv2.imread(image_file)
    structure_res, elapse = table_structurer(img)

    print("result: {}".format(structure_res))