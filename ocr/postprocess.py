import copy
import numpy as np
import cv2
import ocr
import string
from shapely.geometry import Polygon
import pyclipper


def build_post_process(config, global_config=None):
    
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)
    module_class = eval(module_name)(**config)
    return module_class


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.score_mode = score_mode
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict['maps']
        if isinstance(pred, ocr.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch

class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = [
            'ch', 'en', 'EN_symbol', 'french', 'german', 'japan', 'korean',
            'it', 'xi', 'pu', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs', 'oc',
            'rsc', 'bg', 'uk', 'be', 'te', 'ka', 'chinese_cht', 'hi', 'mr',
            'ne', 'EN', 'latin', 'arabic', 'cyrillic', 'devanagari'
        ]
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, character_type)

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "EN_symbol":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        elif character_type in support_character_type:
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(
                character_type)
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)

        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, ocr.Tensor):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
    
class ClsPostProcess(object):
    """ Convert between text-label and text-index """

    def __init__(self, label_list, **kwargs):
        super(ClsPostProcess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, ocr.Tensor):
            preds = preds.numpy()
        pred_idxs = preds.argmax(axis=1)
        decode_out = [(self.label_list[idx], preds[i, idx])
                      for i, idx in enumerate(pred_idxs)]
        if label is None:
            return decode_out
        label = [(self.label_list[idx], 1.0) for idx in label]
        return decode_out, label
    
class TableLabelDecode(object):
    """  """

    def __init__(self,
                 character_dict_path,
                 **kwargs):
        list_character, list_elem = self.load_char_elem_dict(character_dict_path)
        list_character = self.add_special_char(list_character)
        list_elem = self.add_special_char(list_elem)
        self.dict_character = {}
        self.dict_idx_character = {}
        for i, char in enumerate(list_character):
            self.dict_idx_character[i] = char
            self.dict_character[char] = i
        self.dict_elem = {}
        self.dict_idx_elem = {}
        for i, elem in enumerate(list_elem):
            self.dict_idx_elem[i] = elem
            self.dict_elem[elem] = i

    def load_char_elem_dict(self, character_dict_path):
        list_character = []
        list_elem = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            substr = lines[0].decode('utf-8').strip("\n").strip("\r\n").split("\t")
            character_num = int(substr[0])
            elem_num = int(substr[1])
            for cno in range(1, 1 + character_num):
                character = lines[cno].decode('utf-8').strip("\n").strip("\r\n")
                list_character.append(character)
            for eno in range(1 + character_num, 1 + character_num + elem_num):
                elem = lines[eno].decode('utf-8').strip("\n").strip("\r\n")
                list_elem.append(elem)
        return list_character, list_elem

    def add_special_char(self, list_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        list_character = [self.beg_str] + list_character + [self.end_str]
        return list_character

    def __call__(self, preds):
        structure_probs = preds['structure_probs']
        loc_preds = preds['loc_preds']
        if isinstance(structure_probs,ocr.Tensor):
            structure_probs = structure_probs.numpy()
        if isinstance(loc_preds,ocr.Tensor):
            loc_preds = loc_preds.numpy()
        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)
        structure_str, structure_pos, result_score_list, result_elem_idx_list = self.decode(structure_idx,
                                                                                            structure_probs, 'elem')
        res_html_code_list = []
        res_loc_list = []
        batch_num = len(structure_str)
        for bno in range(batch_num):
            res_loc = []
            for sno in range(len(structure_str[bno])):
                text = structure_str[bno][sno]
                if text in ['<td>', '<td']:
                    pos = structure_pos[bno][sno]
                    res_loc.append(loc_preds[bno, pos])
            res_html_code = ''.join(structure_str[bno])
            res_loc = np.array(res_loc)
            res_html_code_list.append(res_html_code)
            res_loc_list.append(res_loc)
        return {'res_html_code': res_html_code_list, 'res_loc': res_loc_list, 'res_score_list': result_score_list,
                'res_elem_idx_list': result_elem_idx_list,'structure_str_list':structure_str}

    def decode(self, text_index, structure_probs, char_or_elem):
        """convert text-label into text-index.
        """
        if char_or_elem == "char":
            current_dict = self.dict_idx_character
        else:
            current_dict = self.dict_idx_elem
            ignored_tokens = self.get_ignored_tokens('elem')
            beg_idx, end_idx = ignored_tokens

        result_list = []
        result_pos_list = []
        result_score_list = []
        result_elem_idx_list = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            elem_pos_list = []
            elem_idx_list = []
            score_list = []
            for idx in range(len(text_index[batch_idx])):
                tmp_elem_idx = int(text_index[batch_idx][idx])
                if idx > 0 and tmp_elem_idx == end_idx:
                    break
                if tmp_elem_idx in ignored_tokens:
                    continue

                char_list.append(current_dict[tmp_elem_idx])
                elem_pos_list.append(idx)
                score_list.append(structure_probs[batch_idx, idx])
                elem_idx_list.append(tmp_elem_idx)
            result_list.append(char_list)
            result_pos_list.append(elem_pos_list)
            result_score_list.append(score_list)
            result_elem_idx_list.append(elem_idx_list)
        return result_list, result_pos_list, result_score_list, result_elem_idx_list

    def get_ignored_tokens(self, char_or_elem):
        beg_idx = self.get_beg_end_flag_idx("beg", char_or_elem)
        end_idx = self.get_beg_end_flag_idx("end", char_or_elem)
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end, char_or_elem):
        if char_or_elem == "char":
            if beg_or_end == "beg":
                idx = self.dict_character[self.beg_str]
            elif beg_or_end == "end":
                idx = self.dict_character[self.end_str]
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of char" \
                              % beg_or_end
        elif char_or_elem == "elem":
            if beg_or_end == "beg":
                idx = self.dict_elem[self.beg_str]
            elif beg_or_end == "end":
                idx = self.dict_elem[self.end_str]
            else:
                assert False, "Unsupport type %s in get_beg_end_flag_idx of elem" \
                              % beg_or_end
        else:
            assert False, "Unsupport type %s in char_or_elem" \
                          % char_or_elem
        return idx

    