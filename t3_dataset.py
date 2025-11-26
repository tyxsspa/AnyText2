'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
import numpy as np
import cv2
import random
import math
import time
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from dataset_util import load, show_bbox_on_image
from opencc import OpenCC
SHOW_GLYPH = False

cc = OpenCC('t2s')

phrase_list = [
    'Text says ',
    'Image with words ',
    'The picture reads ',
    'Captions are ',
    'Texts are ',
    'Text: '
    ' '
]

default_color = [500, 500, 500]

fix_masked_img_bug = True


def tra_chinese(text):
    simplified_text = cc.convert(text)
    return text != simplified_text


def copy_and_rename_dict_keys(original_dict, key_mapping):
    new_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in original_dict:
            new_dict[new_key] = original_dict[old_key]
    return new_dict


key_mapping = {
    'en': 'Latin',
    'ch_sim_char': 'Chinese_sim',
    'ch_tra_char': 'Chinese_tra',
    'hi': 'Hindi',
    'ar': 'Arabic',
    'ja': 'Japanese',
    'ko': 'Korean',
    'bn': 'Bangla',
}


def random_rotate(image, angle_range):
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def random_translate(image, translate_range):
    tx = random.uniform(translate_range[0], translate_range[1])
    ty = random.uniform(translate_range[0], translate_range[1])
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    h, w = image.shape[:2]
    translated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return translated


def random_scale(image, scale_range):
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    scaled = cv2.resize(image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    if scale >= 1:
        scaled = scaled[(scaled.shape[0]-h)//2: (scaled.shape[0]+h)//2, (scaled.shape[1]-w)//2: (scaled.shape[1]+w)//2]
    else:
        pad_h = (h - scaled.shape[0]) // 2
        pad_w = (w - scaled.shape[1]) // 2
        scaled = cv2.copyMakeBorder(scaled, pad_h, h - scaled.shape[0] - pad_h, pad_w, w - scaled.shape[1] - pad_w, cv2.BORDER_REPLICATE)
    return scaled


def random_augment(image, rot=(-10, 10), trans=(-5, 5), scale=(0.9, 1.1)):
    image = random_rotate(image, rot)
    image = random_translate(image, trans)
    image = random_scale(image, scale)
    return image


def insert_spaces(text, num_spaces):
    return (' ' * num_spaces).join(text)


def draw_glyph(font, text):
    if isinstance(font, str):
        font = ImageFont.truetype(font, size=60)
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right-left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W*0.9/text_width, H*0.9/text_height)
    new_font = font.font_variant(size=int(g_size*ratio))

    text_width, text_height = new_font.getsize(text)
    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y//2
    draw.text((x, y), text, font=new_font, fill='white')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
    if SHOW_GLYPH:
        i = 0
        file_path = f'tmp_{i}.jpg'
        while os.path.exists(file_path):
            i += 1
            file_path = f'tmp_{i}.jpg'
        cv2.imwrite(file_path, img*255)
    return img


def draw_glyph2(font, text, polygon, color, vertAng=10, scale=1, width=512, height=512, add_space=True):
    def initialize_img(width, height, scale):
        img = np.zeros((height * scale, width * scale, 3), np.uint8)
        return Image.fromarray(img)

    def prepare_image(img):
        return np.array(img.convert('RGB')).astype(np.float64) / 255.0

    try:
        if color.mean() < 0:
            color = np.array([255, 255, 255])
        color = np.clip(color, 10, 255)  # RGB >= 10
        if isinstance(font, str):
            if os.path.exists(font):
                font = ImageFont.truetype(font, size=60)
            else:
                img = initialize_img(width, height, scale)
                return prepare_image(img)
        enlarge_polygon = np.array(polygon) * scale
        rect = cv2.minAreaRect(enlarge_polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w, h = rect[1]
        angle = rect[2]

        if angle < -45:
            angle += 90
        angle = -angle
        if w < h:
            angle += 90

        vert = False
        if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
            _w = max(box[:, 0]) - min(box[:, 0])
            _h = max(box[:, 1]) - min(box[:, 1])
            if _h >= _w:
                vert = True
                angle = 0

        img = initialize_img(width, height, scale)
        image4ratio = Image.new("RGB", img.size, "white")
        draw = ImageDraw.Draw(image4ratio)
        min_dim = min(w, h)
        max_dim = max(w, h)

        # Binary search for optimal font size
        def adjust_font_size(min_size, max_size, text):
            while min_size < max_size:
                mid_size = (min_size + max_size) // 2
                new_font = font.font_variant(size=int(mid_size))
                bbox = draw.textbbox((0, 0), text=text, font=new_font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                if text_w <= max_dim and text_h <= min_dim:
                    min_size = mid_size + 1
                else:
                    max_size = mid_size
            return max_size - 1

        optimal_font_size = adjust_font_size(1, min_dim, text)
        new_font = font.font_variant(size=int(optimal_font_size))

        extra_space = 0
        if add_space:
            if vert:
                # Calculate total height with added space
                total_height = sum(draw.textbbox((0, 0), text=char, font=new_font)[3] -
                                   draw.textbbox((0, 0), text=char, font=new_font)[1]
                                   for char in text)
                if total_height < max_dim and len(text) > 1:
                    extra_space = (max_dim - total_height) // (len(text) - 1)
            else:
                # Handle horizontal text space addition
                for i in range(1, 100):
                    text_space = insert_spaces(text, i)
                    bbox2 = draw.textbbox((0, 0), text=text_space, font=new_font)
                    text_w, text_h = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
                    if text_w > max_dim or text_h > min_dim:
                        text = insert_spaces(text, i - 1)
                        break

        left, top, right, bottom = draw.textbbox((0, 0), text=text, font=new_font)
        text_width = right - left
        text_height = bottom - top

        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        if not vert:
            text_y_center = rect[0][1] - (text_height / 2)
            draw.text((rect[0][0] - text_width / 2, text_y_center - top), text, font=new_font, fill=tuple(color)+(255,))
        else:
            x_s = min(box[:, 0]) + _w // 2 - text_height // 2
            y_s = min(box[:, 1]) - top
            for c in text:
                draw.text((x_s, y_s), c, font=new_font, fill=tuple(color)+(255,))
                _, _t, _, _b = draw.textbbox((0, 0), text=c, font=new_font)
                char_height = _b - _t
                y_s += char_height + extra_space

        rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))
        x_offset = int((img.width - rotated_layer.width) / 2)
        y_offset = int((img.height - rotated_layer.height) / 2)
        img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)

        return prepare_image(img)

    except Exception as e:
        print(f"An error occurred in draw_glyph2: {e}")
        img = initialize_img(width, height, scale)
        return prepare_image(img)


'''
target_img: (-1,1), hwc
return font_hint: (0,1), hw1
'''


def draw_font_hint(target_img, polygon, target_area_range=[1.0, 1.0], prob=1.0, randaug=False):
    height, width, _ = target_img.shape
    img = np.zeros((height, width), dtype=np.uint8)

    if random.random() < (1 - prob):  # Empty font hint
        return img[..., None] / 255.0, img[..., None] / 255.0

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(img, [pts], color=255)
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    rect_width, rect_height = size
    x, y, w, h = cv2.boundingRect(np.clip(polygon, 0, None))
    target_img_scaled = (target_img + 1.0) / 2.0
    cropped_ori_img = target_img_scaled[y:y+h, x:x+w]
    if randaug:
        augmented_cropped = random_augment(cropped_ori_img, rot=(-10, 10), trans=(-10, 10), scale=(0.9, 1.1))
    else:
        augmented_cropped = cropped_ori_img
    augmented_cropped_gray = cv2.cvtColor((augmented_cropped * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    thresholded = cv2.adaptiveThreshold(augmented_cropped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresholded_resized = np.zeros_like(img.squeeze())
    thresholded_resized[y:y+h, x:x+w] = (1 - thresholded / 255.0)

    # gen a random mask
    area_ratio = random.uniform(target_area_range[0], target_area_range[1])
    long_side, short_side = max(rect_width, rect_height), min(rect_width, rect_height)
    long_axis_mask_length = long_side * (1 - area_ratio)
    angle_rad = np.radians(angle)
    rect_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    if rect_width < rect_height:
        rect_vector = np.array([-rect_vector[1], rect_vector[0]])
    start_offset = random.uniform(0, long_side - long_axis_mask_length)
    start_point = center - rect_vector * (long_side / 2 - start_offset)
    mask_center = start_point + rect_vector * (long_axis_mask_length / 2)
    mask_vector = rect_vector * (long_axis_mask_length / 2)
    short_axis_vector = np.array([-rect_vector[1], rect_vector[0]]) * (short_side / 2)
    mask_corners = np.array([
        mask_center - mask_vector - short_axis_vector,
        mask_center + mask_vector - short_axis_vector,
        mask_center + mask_vector + short_axis_vector,
        mask_center - mask_vector + short_axis_vector
    ], dtype=np.int32)
    cv2.fillPoly(img, [mask_corners], color=0)
    img = img[..., None] / 255.0

    # Compute font hint
    font_hint = img.squeeze() * thresholded_resized
    return font_hint[..., None], img


def get_text_caption(n_line, ori_caption, place_holder='*'):
    return random.choice(phrase_list) + ' , '.join([f'{place_holder}']*n_line) + ' . '


def generate_random_rectangles(w, h, box_num):
    rectangles = []
    for i in range(box_num):
        x = random.randint(0, w)
        y = random.randint(0, h)
        w = random.randint(16, 256)
        h = random.randint(16, 96)
        angle = random.randint(-45, 45)
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)
        p1 = rotate_point(p1, center, angle)
        p2 = rotate_point(p2, center, angle)
        p3 = rotate_point(p3, center, angle)
        p4 = rotate_point(p4, center, angle)
        rectangles.append((p1, p2, p3, p4))
    return rectangles


def rotate_point(point, center, angle):
    # rotation
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


def truncate_string(input_string, max_words=45):
    input_string = input_string[:500]  # be careful for repeat CN words, may cause OOM
    words = input_string.split()
    if len(words) <= max_words:
        return input_string
    # Split into sentences
    sentences = input_string.split('.')
    first_sentence = sentences[0].strip()
    remaining_sentences = sentences[1:]
    # Start with the first sentence
    truncated_string = first_sentence
    words_count = len(first_sentence.split())
    if words_count >= max_words:
        return ' '.join(first_sentence.split()[:max_words])
    # Randomly add sentences until we reach or exceed the word limit
    while words_count < max_words and remaining_sentences:
        random_index = random.randint(0, len(remaining_sentences) - 1)
        sentence = remaining_sentences.pop(random_index).strip()
        sentence_words = sentence.split()
        if words_count + len(sentence_words) > max_words:
            # If adding the sentence exceeds word limit, truncate the sentence
            sentence_words = sentence_words[:max_words - words_count] + ['.']
        if sentence_words:
            truncated_string += '. ' + ' '.join(sentence_words)
            words_count += len(sentence_words)
    return truncated_string


class T3DataSet(Dataset):
    def __init__(
            self,
            json_path,
            max_lines=5,
            max_chars=20,
            place_holder='*',
            font_path='./font/Arial_Unicode.ttf',
            mask_pos_prob=1.0,
            mask_img_prob=0.5,
            for_show=False,
            using_dlc=False,
            glyph_scale=1,
            percent=1.0,
            debug=False,
            wm_thresh=1.0,
            render_glyph=True,
            trunc_cap=128,  # caption truncation
            rand_font=False,
            lang_font_path='./font/lang_font_dict.npy',
            font_hint_prob=0,  # set 0 to disable font_hint
            color_prob=1.0,  # set 0 to disable color
            font_hint_area=[1.0, 1.0],  # reserved area on each font_hint line
            font_hint_randaug=False,
            cap_watermark=True,
            img_wh=512,
            ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        data_list = []
        self.using_dlc = using_dlc
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.place_holder = place_holder
        self.font = ImageFont.truetype(font_path, size=60)
        self.mask_pos_prob = mask_pos_prob
        self.mask_img_prob = mask_img_prob
        self.for_show = for_show
        self.glyph_scale = glyph_scale
        self.wm_thresh = wm_thresh
        self.render_glyph = render_glyph
        self.trunc_cap = trunc_cap
        self.rand_font = rand_font
        self.font_hint_prob = font_hint_prob
        self.color_prob = color_prob
        self.font_hint_area = font_hint_area
        self.font_hint_randaug = font_hint_randaug
        self.cap_watermark = cap_watermark
        self.img_wh = img_wh

        if self.rand_font:
            lang_font_dict = np.load(lang_font_path, allow_pickle=True)[()]
            self.lang_font = copy_and_rename_dict_keys(lang_font_dict, key_mapping)
            for lang in self.lang_font:
                self.lang_font[lang] = [ImageFont.truetype(p, size=60) for p in self.lang_font[lang]['fonts']]
            print('rand_font=True, all fonts are loaded!')

        for jp in json_path:
            data_list += self.load_data(jp, percent)
        self.data_list = data_list
        print(f'All dataset loaded, imgs={len(self.data_list)}')
        self.debug = debug
        if self.debug:
            self.tmp_items = [i for i in range(100)]

    def load_data(self, json_path, percent):
        tic = time.time()
        content = load(json_path)
        d = []
        count = 0
        wm_skip = 0
        max_img = len(content['data_list']) * percent
        for gt in content['data_list']:
            if len(d) > max_img:
                break
            if 'wm_score' in gt and gt['wm_score'] > self.wm_thresh:  # wm_score > thresh will be skiped as an img with watermark
                wm_skip += 1
                continue
            data_root = content['data_root']
            if self.using_dlc:
                data_root = data_root.replace('/data/vdb', '/mnt/data', 1)
            img_path = os.path.join(data_root, gt['img_name'])
            info = {}
            info['img_path'] = img_path
            info['caption'] = gt['caption'] if 'caption' in gt else ''
            if 'wm_score' in gt and self.cap_watermark:
                if gt['wm_score'] > 0.5:
                    info['caption'] += ' with watermarks '
                else:
                    info['caption'] += ' no watermarks '
            if self.place_holder in info['caption']:
                count += 1
                info['caption'] = info['caption'].replace(self.place_holder, " ")
            if 'annotations' in gt:
                polygons = []
                invalid_polygons = []
                texts = []
                languages = []
                pos = []
                color = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0:
                        continue
                    if 'valid' in annotation and annotation['valid'] is False:
                        invalid_polygons.append(annotation['polygon'])
                        continue
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    lang = annotation['language']
                    if lang == 'Chinese':
                        lang = 'Chinese_tra' if tra_chinese(annotation['text']) else 'Chinese_sim'
                    languages.append(lang)
                    if 'pos' in annotation:
                        pos.append(annotation['pos'])
                    if 'color' in annotation:
                        color.append(annotation['color'])
                    else:
                        color.append(default_color)
                info['polygons'] = [np.array(i) for i in polygons]
                info['invalid_polygons'] = [np.array(i) for i in invalid_polygons]
                info['texts'] = texts
                info['language'] = languages
                info['pos'] = pos
                info['color'] = [np.array(i) for i in color]
            d.append(info)
        print(f'{json_path} loaded, imgs={len(d)}, wm_skip={wm_skip}, time={(time.time()-tic):.2f}s')
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: {self.place_holder}, change to ' '...")
        return d

    def __getitem__(self, item):
        editing_mode = False
        if random.random() < self.mask_img_prob:
            editing_mode = True
        item_dict = {}
        if self.debug:  # sample fixed items
            item = self.tmp_items.pop()
            print(f'item = {item}')
        cur_item = self.data_list[item]
        # img
        target = np.array(Image.open(cur_item['img_path']).convert('RGB'))
        if target.shape[0] != self.img_wh or target.shape[1] != self.img_wh:
            target = cv2.resize(target, (self.img_wh, self.img_wh))
        target = (target.astype(np.float32) / 127.5) - 1.0
        item_dict['img'] = target
        # caption
        if self.trunc_cap > 0:
            cur_item['caption'] = truncate_string(cur_item['caption']+'. ', max_words=self.trunc_cap)
        item_dict['img_caption'] = cur_item['caption']
        item_dict['text_caption'] = ''
        item_dict['glyphs'] = []
        item_dict['gly_line'] = []
        item_dict['positions'] = []
        font_hints = []
        font_hints_mask = []

        item_dict['texts'] = []
        item_dict['language'] = []
        item_dict['inv_mask'] = []
        item_dict['color'] = []
        texts = cur_item.get('texts', [])
        if len(texts) == 0:  # padding empty text on image, prevent mismatch between collectives on ranks
            texts = [' ']
            cur_item['color'] = [np.array(default_color)]
            cur_item['polygons'] = [np.array([[10, 10], [100, 10], [100, 100], [10, 100]])]
            cur_item['language'] = ['Latin']
            cur_item['texts'] = texts

        idxs = [i for i in range(len(texts))]
        if len(texts) > self.max_lines:
            sel_idxs = random.sample(idxs, self.max_lines)
            unsel_idxs = [i for i in idxs if i not in sel_idxs]
        else:
            sel_idxs = idxs
            unsel_idxs = []
        item_dict['color'] = [cur_item['color'][i] for i in sel_idxs]
        if self.color_prob < 1.0:
            for i, c in enumerate(item_dict['color']):
                if random.random() < (1 - self.color_prob):
                    item_dict['color'][i] = np.array(default_color)
        item_dict['text_caption'] = get_text_caption(len(sel_idxs), self.place_holder)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:self.max_chars] for i in sel_idxs]
        item_dict['language'] = [cur_item['language'][i] for i in sel_idxs]
        # glyphs
        for idx, text in enumerate(item_dict['texts']):
            if self.rand_font:
                lang = item_dict['language'][idx]
                assert lang in self.lang_font
                use_font = random.choice(self.lang_font[lang])  # random font
            else:
                use_font = self.font  # arial unicode
            gly_line = draw_glyph(use_font, text)
            if self.render_glyph:
                glyphs = draw_glyph2(use_font, text, item_dict['polygons'][idx], item_dict['color'][idx], scale=self.glyph_scale, width=self.img_wh, height=self.img_wh)
            else:
                glyphs = np.zeros((self.img_wh*self.glyph_scale, self.img_wh*self.glyph_scale, 3), np.float64)
            item_dict['glyphs'] += [glyphs]
            item_dict['gly_line'] += [gly_line]
        # mask_pos
        for polygon in item_dict['polygons']:
            target_area_ratio_pos = [1.0, 1.0]  # 0.6--0.9
            item_dict['positions'] += [self.draw_pos(polygon, self.mask_pos_prob, target_area_ratio_pos)]
            if self.font_hint_prob > 0:
                font_hint, font_hint_mask = draw_font_hint(target, polygon, target_area_range=self.font_hint_area, prob=self.font_hint_prob, randaug=self.font_hint_randaug)
                font_hints += [font_hint]
                font_hints_mask += [font_hint_mask]

        # inv_mask
        invalid_polygons = cur_item['invalid_polygons'] if 'invalid_polygons' in cur_item else []
        if len(texts) > 0:
            invalid_polygons += [cur_item['polygons'][i] for i in unsel_idxs]
        if editing_mode:
            # randomly generate 0 masks (disabled)
            box_num = random.randint(0, 0)
            boxes = generate_random_rectangles(self.img_wh, self.img_wh, box_num)
            boxes = np.array(boxes)
            pos_list = item_dict['positions'].copy()
            for i in range(box_num):
                pos_list += [self.draw_pos(boxes[i], self.mask_pos_prob)]
            invalid_polygons = []  # clear invalid_polygons for editing mode
            mask = self.get_hint(pos_list)
            if fix_masked_img_bug:
                masked_img = (target-mask*10).clip(-1, 1)
            else:
                masked_img = target*(1-mask)
        else:
            if fix_masked_img_bug:
                masked_img = np.zeros_like(target)-1
            else:
                masked_img = np.zeros_like(target)
        item_dict['masked_img'] = masked_img
        item_dict['inv_mask'] = self.draw_inv_mask(invalid_polygons)
        item_dict['hint'] = self.get_hint(item_dict['positions'])
        item_dict['font_hint'] = self.get_hint(font_hints)

        if self.for_show:
            item_dict['img_name'] = os.path.split(cur_item['img_path'])[-1]
            return item_dict
        if len(texts) > 0:
            del item_dict['polygons']
        # padding
        n_lines = min(len(texts), self.max_lines)
        item_dict['n_lines'] = n_lines
        n_pad = self.max_lines - n_lines
        if n_pad > 0:
            item_dict['glyphs'] += [np.zeros((self.img_wh*self.glyph_scale, self.img_wh*self.glyph_scale, 3))] * n_pad
            item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
            item_dict['positions'] += [np.zeros((self.img_wh, self.img_wh, 1))] * n_pad
            item_dict['texts'] += [' '] * n_pad
            item_dict['language'] += [' '] * n_pad
            item_dict['color'] += [np.array(default_color)] * n_pad

        return item_dict

    def __len__(self):
        return len(self.data_list)

    def draw_inv_mask(self, polygons):
        img = np.zeros((self.img_wh, self.img_wh))
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img/255.

    def draw_pos(self, ploygon, prob=1.0, target_area_range=[1.0, 1.0]):
        img = np.zeros((self.img_wh, self.img_wh))
        rect = cv2.minAreaRect(ploygon)
        center, size, angle = rect
        w, h = size
        small = False
        min_wh = 20*self.img_wh/512
        if w < min_wh or h < min_wh:
            small = True
        if random.random() < prob:
            pts = ploygon.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
            # 10% dilate / 10% erode / 5% dilatex2  5% erodex2
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7:
                pass
            elif random_value < 0.8:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.9 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.95:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
            elif random_value < 1.0 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
            # gen a random mask(for editing mode)
            if target_area_range[0] < 1.0 or target_area_range[1] < 1.0:
                area_ratio = random.uniform(target_area_range[0], target_area_range[1])
                long_side, short_side = max(w, h), min(w, h)
                long_axis_mask_length = long_side * (1 - area_ratio)
                angle_rad = np.radians(angle)
                rect_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
                if w < h:
                    rect_vector = np.array([-rect_vector[1], rect_vector[0]])
                start_offset = random.uniform(0, long_side - long_axis_mask_length)
                start_point = center - rect_vector * (long_side / 2 - start_offset)
                mask_center = start_point + rect_vector * (long_axis_mask_length / 2)
                mask_vector = rect_vector * (long_axis_mask_length / 2)
                short_axis_vector = np.array([-rect_vector[1], rect_vector[0]]) * (short_side / 2)
                mask_corners = np.array([
                    mask_center - mask_vector - short_axis_vector,
                    mask_center + mask_vector - short_axis_vector,
                    mask_center + mask_vector + short_axis_vector,
                    mask_center - mask_vector + short_axis_vector
                ], dtype=np.int32)
                cv2.fillPoly(img, [mask_corners], color=0)
        img = img[..., None]
        return img/255.

    def get_hint(self, positions):
        if len(positions) == 0:
            return np.zeros((self.img_wh, self.img_wh, 1))
        return np.sum(positions, axis=0).clip(0, 1)


if __name__ == '__main__':
    '''
    Run this script to show details of your dataset, such as ocr annotations, glyphs, prompts, etc.
    '''
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import shutil

    show_imgs_dir = 'show_results'
    show_count = 100
    if os.path.exists(show_imgs_dir):
        shutil.rmtree(show_imgs_dir)
    os.makedirs(show_imgs_dir)
    plt.rcParams['axes.unicode_minus'] = False
    json_paths = [
        '/path/of/your/dataset/data1.json',
        '/path/of/your/dataset/data2.json',
        # ...
    ]
    dataset = T3DataSet(json_paths, for_show=True, max_lines=20, glyph_scale=1, mask_img_prob=0.5,
                        render_glyph=True, rand_font=True, font_hint_prob=1, font_hint_area=[0.7, 1],
                        font_hint_randaug=True, color_prob=1, img_wh=512)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    pbar = tqdm(total=show_count)
    for i, data in enumerate(train_loader):
        if i == show_count:
            break
        img = ((data['img'][0].numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        masked_img = ((data['masked_img'][0].numpy() + 1.0) / 2.0 * 255)[..., ::-1].astype(np.uint8)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_masked.jpg'), masked_img)
        if 'texts' in data and len(data['texts']) > 0:
            texts = [x[0] for x in data['texts']]
            img = show_bbox_on_image(Image.fromarray(img), data['polygons'], texts)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}.jpg'),  np.array(img)[..., ::-1])
        with open(os.path.join(show_imgs_dir, f'plots_{i}.txt'), 'w') as fin:
            fin.writelines([data['img_caption'][0], data['text_caption'][0]])
        all_glyphs = []
        for k, glyphs in enumerate(data['glyphs']):
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_glyph_{k}.jpg'), (glyphs[0].numpy()*255).astype(np.int32)[..., ::-1])
            all_glyphs += [(glyphs[0].numpy()*255).astype(np.int32)[..., ::-1]]
        if len(all_glyphs) > 0:
            img_allglyph = np.sum(all_glyphs, axis=0)
            black_pixels = (img_allglyph[:, :, 0] < 5) & (img_allglyph[:, :, 1] < 5) & (img_allglyph[:, :, 2] < 5)
            img_allglyph[black_pixels] = [215, 215, 215]
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_allglyphs.jpg'), img_allglyph)
        for k, gly_line in enumerate(data['gly_line']):
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_gly_line_{k}.jpg'), gly_line[0].numpy().astype(np.int32)*255)
        for k, position in enumerate(data['positions']):
            if position is not None:
                cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_pos_{k}.jpg'), position[0].numpy().astype(np.int32)*255)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_hint.jpg'), data['hint'][0].numpy().astype(np.int32)*255)
        img_font_hint = (1-data['font_hint'][0].numpy()).astype(np.int32)*215
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_font_hint.jpg'), img_font_hint)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_inv_mask.jpg'), np.array(img)[..., ::-1]*(1-data['inv_mask'][0].numpy().astype(np.int32)))
        pbar.update(1)
    pbar.close()
