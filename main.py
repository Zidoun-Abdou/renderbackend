from readmrz import MrzDetector, MrzReader
import pytesseract
import random
from fastapi import UploadFile, File
import shutil
import cv2
import codecs
import re
import numpy as np
import io
import base64
from datetime import datetime
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw,ImageFont
from bidi.algorithm import get_display
from typing import Optional
from io import BytesIO


app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tesseract_path = '/usr/bin/tesseract'
pytesseract = pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path

detector = MrzDetector()
reader = MrzReader()



@app.post("/mrz")
async def root(myphoto: UploadFile = File(...)):
    try:
        dict_recult={}
        original_frant = "path_frant.png"
        with open(original_frant, "wb") as f:
            shutil.copyfileobj(myphoto.file, f)
        image = detector.read(original_frant)
        cropped = detector.crop_area(image)
        result = reader.process(cropped)

        if int(result['birth_date'][0:2])>10:
            dict_recult["birth_date"]= result['birth_date'][2:4]+'/'+result['birth_date'][4:]+'/'+"19"+result['birth_date'][0:2]
        else:
            dict_recult["birth_date"] = result['birth_date'][2:4]+'/'+result['birth_date'][4:]+'/'+"20"+result['birth_date'][0:2]

        dict_recult["expiry_date"] = result['expiry_date'][2:4] + '/' + result['expiry_date'][4:] + '/' + "20" + result['expiry_date'][0:2]
        dict_recult["document_number"]=result["document_number"]
        return {
            "success": True,
            "mrz":dict_recult
        }

    except:
        return {
            "success": False,
        }

@app.get("/")
async def root():
    return {"message": "Hello Worldss"}





@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}




def base64_to_image(base64_string, target_size=(300, 300)):
    # Remove the data URI prefix if it exists
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]

    # Decode the Base64 string
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object to read the image data
    image_stream = BytesIO(image_data)

    # Open the image using PIL (Pillow)
    image = Image.open(image_stream)

    # Resize the image
    resized_image = image.resize(target_size)

    return resized_image

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Encode the binary data of the image using base64
        encoded_string = base64.b64encode(image_file.read())
        # Convert bytes to a string
        base64_string = encoded_string.decode("utf-8")
        return base64_string

@app.post("/generate_image/")
async def generate_image(
    n_appel: str = Form(...),
    n_carte_sim: str = Form(...),
    n_serie: str = Form(...),
    imei: str = Form(...),
    daira: str = Form(...),
    baladia_latin: str = Form(...),
    baladia_arab: str = Form(...),
    deliv_date: str = Form(...),
    exp_date: str = Form(""),
    surname_latin: str = Form(...),
    surname_arabic: str = Form(...),
    birthplace_latin: str = Form(""),
    birthplace_arabic: str = Form(""),
    birth_date: str = Form(""),
    sexe_latin: str = Form(""),
    sex_arabic: str = Form(""),
    blood_type: str = Form(""),
    nin: str = Form(""),
    name_latin: str = Form(""),
    name_arabic: str = Form(""),
    base64_text: str = Form(""),
        base64_face: str = Form(""),

):
    try:
        img = Image.open("contrat.jpg")
        draw = ImageDraw.Draw(img)

        # Load a font
        # You can replace "arial.ttf" with the path to your desired font file
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 80, encoding="utf-8")
        # Set the text color
        text_color = (0, 0, 0)

        # Draw the text on the image
        img.paste(base64_to_image(base64_face, target_size=(300, 300)), (2150, 300))
        draw.text((1000, 650), n_appel, font=font, fill=text_color)
        draw.text((1000, 755), n_carte_sim, font=font, fill=text_color)
        draw.text((300, 860), n_serie, font=font, fill=text_color)
        draw.text((1400, 860), imei, font=font, fill=text_color)
        draw.text((280, 1200), surname_latin, font=font, fill=text_color)
        draw.text((1500, 1200), surname_arabic, font=font, fill=text_color)
        draw.text((280, 1305), name_latin, font=font, fill=text_color)
        draw.text((1500, 1305), name_arabic, font=font, fill=text_color)
        draw.text((1000, 1410), birth_date, font=font, fill=text_color)
        draw.text((1600, 1715), birthplace_arabic, font=font, fill=text_color)
        draw.text((280, 1715), birthplace_latin, font=font, fill=text_color)
        draw.text((1000, 1820), daira, font=font, fill=text_color)
        draw.text((445, 2115), "X", font=font, fill=text_color)
        draw.text((1000, 2220), nin, font=font, fill=text_color)
        draw.text((280, 2325), deliv_date, font=font, fill=text_color)
        draw.text((1610, 2325), baladia_latin, font=font, fill=text_color)
        # Get the current date
        today = datetime.today().strftime('%d/%m/%Y')
        draw.text((280, 2795), today, font=font, fill=text_color)


        # Paste the photo onto the main image
        img.paste(base64_to_image(base64_text, target_size=(500, 300)), (1310, 2795))


        # Save the modified image
        output_path = 'output.jpg'
        img.save(output_path)

        image_path = "output.jpg"
        base64_string = image_to_base64(image_path)

        return {
      "success": True,
      "image": base64_string
          }
    except:
        return {
            "success": False,
        }


class LibraryLoader:
    _loaded = False

    def load_libraries(self):
        if not LibraryLoader._loaded:
            # load libraries
            import easyocr, torch
            # from passporteye import read_mrz
            # from ArabicOcr import arabicocr
            from datetime import datetime
            import re, os, pytesseract
            from pathlib import Path
            # from passporteye import read_mrz as passport_mrz_reader

            updated_string = str(Path.cwd())
            updated_string = updated_string.replace("\\", "/")
            self.tesseract_path = '/usr/bin/tesseract'
            self.re = re
            self.os = os
            self.pytesseract = pytesseract
            self.datetime = datetime
            self.tessetact_congfig = "tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ########################################## Path of card detection model and weights ###########################################################
            path_model_card_detection = updated_string + '/cardp/yolov5'
            path_model_card_detection_weights = updated_string + '/cardp/yolov5/runs/train/exp3the_bestfinal/weights/best.pt'
            ########################################## Path of card information detection model and weights ###########################################################
            path_model_info_card = updated_string + '/card_info/yolov5'
            path_model_info_card_weights = updated_string + '/card_info/yolov5/runs/train/exp47bestv1/weights/best.pt'
            # load models for just one time
            self.model_card_detection = torch.hub.load(path_model_card_detection, 'custom',
                                                       path_model_card_detection_weights, source='local')
            self.model_card_detection.conf = 0.92
            self.model_info_extraction = torch.hub.load(path_model_info_card, 'custom', path_model_info_card_weights,
                                                        source='local')
            # load Static variabels
            self.path_frant = updated_string + '/images/Front_card.jpg'
            self.path_back = updated_string + '/images/Back_card.jpg'
            # --------------------------------------------- path for saved croped card image ---------------------------------------------------------------
            self.path_save_frant = updated_string + '/saved_images_card_detection/image_resultR.png'
            self.path_save_back = updated_string + '/saved_images_card_detection/image_resultV.png'
            self.reader_easy_ocr = easyocr.Reader(['en'])
            self.nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

            _loaded = True
            print('_loaded:  ', _loaded)


library_loader = LibraryLoader()
library_loader.load_libraries()


def face_card_detection(path, path_save_frant):
    dict_result = {"result": True}
    try:
        results = library_loader.model_card_detection(path)
        crops = results.crop(save=False)
        image = crops[0]['im']
        name = crops[0]['label']
        name, x = name.split(' ')
        if name == 'face_card' and len(image) > 0:
            dict_result.update(face_card=image)
            cv2.imwrite(path_save_frant, image)
            return dict_result
        else:
            dict_result["result"] = False
            return dict_result
    except:
        dict_result["result"] = False
        return dict_result


# -------------------------------------------------------------------------- Frant_card info exctraction function -------------------------------------------------------------------
def info_extraction_frant(path):
    result_dic = {"result": False}
    list_result = ['arabic_name',
                   'arabic_surname',
                   'birth_date',
                   'card_number',
                   'creation_date',
                   'expiration_date',
                   'face_image',
                   'id_number',
                   ]
    try:
        results = library_loader.model_info_extraction(path)
        labels, precision, cord_point = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]
        cpt = 0
        for i in labels:
            crops = results.crop(save=False)
            crops[cpt]['label'], x = crops[cpt]['label'].split(' ')
            if crops[cpt]['label'] in list_result:
                if crops[cpt]['label'] not in result_dic.keys():
                    result_dic['result'] = True
                    result_dic[crops[cpt]['label']] = crops[cpt]['im']
            cpt = cpt + 1
        return result_dic
    except:
        return result_dic


# -------------------------------------------------------------------------------------- Card Frant detection and extraction --------------------------------------------------------------------------------------
def card_frant(path, path_save_frant):
    try:
        result_card_frant = face_card_detection(path, path_save_frant)
        if len(list(result_card_frant.keys())) == 2 and list(result_card_frant.keys())[1] == 'face_card':
            detection_result = info_extraction_frant(path_save_frant)
            return detection_result
        else:
            detection_result = {"result": False}
            return detection_result
    except:
        detection_result = {"result": False}
        return detection_result


# ------------------------------------------------------------------------------------- OCR Functions --------------------------------------------------------------------------------------------------

#####################################  dates functions   ###################################################
class Date():
    def __init__(self, image, class_name) -> None:
        self.image = image
        self.class_name = class_name

    def easy_ocr_func_dates(self):
        try:
            easy_result = library_loader.reader_easy_ocr.readtext_batched(self.image, paragraph=True)[0][-1][-1]
            if easy_result[-1] not in library_loader.nums:
                easy_result = easy_result[:-1]
            easy_result = easy_result.replace(".", "-")
            return easy_result
        except:
            return 'nothing_easyocr'

    def years_between_dates(self, date1, date2):
        return int((date1 - date2).days / 365.25)

    def dates_main(self) -> dict:
        if self.class_name in ['expiration_date', 'birth_date', 'creation_date']:
            easy_ocr_date = self.easy_ocr_func_dates()
            if len(easy_ocr_date) == 10:
                date = library_loader.datetime.strptime(easy_ocr_date, '%Y-%m-%d')
                today = library_loader.datetime.today()
                if self.class_name == 'expiration_date':
                    if date > today:
                        return {'decision': self.class_name, 'result': easy_ocr_date}
                    else:
                        return {'decision': False, 'result': easy_ocr_date}
                elif self.class_name in ['birth_date', 'creation_date']:
                    if date < today:
                        key = 'age' if self.class_name == 'birth_date' else 'delivered'
                        return {'decision': self.class_name, 'result': easy_ocr_date}
                    else:
                        return {'decision': False, 'result': easy_ocr_date}
            else:
                return {'decision': False, 'result': 'bad_image'}
        else:
            return {'decision': False, 'result': 'bad_class_name'}


#####################################  numbers functions   ###################################################
class Numbers():
    def __init__(self, image, class_name) -> None:
        self.image = image
        self.class_name = class_name

    def easy_ocr_func_number(self):
        try:
            easy_result = library_loader.reader_easy_ocr.readtext(self.image)
            conf = easy_result[0][-1]
            easy_result = easy_result[0][-2]

            if easy_result[-1] not in library_loader.nums:
                easy_result = easy_result[:-1]
            return easy_result
        except:
            return 'nothing_easyocr'

    def numbers_main(self) -> dict:
        if self.class_name not in ['id_number', 'card_number']:
            return {'decision': False, 'result': 'bad_class_name'}
        easy_ocr_nums = self.easy_ocr_func_number()
        length = (len(easy_ocr_nums))
        if length not in [9, 18]:
            return {'decision': False, 'result': 'bad_image'}
        if length == 9 and self.class_name == 'card_number':
            return {'decision': self.class_name, 'result': easy_ocr_nums}
        elif length == 18 and self.class_name == 'id_number':
            return {'decision': self.class_name, 'result': easy_ocr_nums}
        else:
            return {'decision': False, 'result': 'BAD_CLASS_NAME'}


# -------------------------------------------------------------------------------------------- Remouve images function ---------------------------------------------------------------------------------------------------------
def remouve_images(original_frant, path_save_frant):
    try:
        library_loader.os.remove("arabe.jpg")
    except:
        pass
    try:
        library_loader.os.remove("arab_dates.png")
    except:
        pass
    try:
        library_loader.os.remove('arab_number.jpg')
    except:
        pass
    try:
        library_loader.os.remove('arabe_numbers.png')
    except:
        pass
    try:
        library_loader.os.remove(original_frant)
    except:
        pass
    try:
        library_loader.os.remove(path_save_frant)
    except:
        pass


# ----------------------------------------------------------------------------- save image cards Frant and back to start the predicttions ----------------------------------------------------------------------------------------
def save_image_card(original_frant, original_back, Frant_image_card, Back_image_card):
    cv2.imwrite(original_frant, Frant_image_card)
    cv2.imwrite(original_back, Back_image_card)


# -------------------------------------------------------------------------------------------- API Whowiyati ----------------------------------------------------------------------------------------
@app.post("/Whowiyati_KYC/")
async def Whowiyati_KYC(front_image: UploadFile = File(...)):
    try:
        path_save_frant = library_loader.path_save_frant
        original_frant = library_loader.path_frant
        with open(original_frant, "wb") as f:
            shutil.copyfileobj(front_image.file, f)
        random_float = random.uniform(0, 1)
        random_float = str(random_float)
        dict_info_frant = card_frant(original_frant, path_save_frant)
        result = {'decision': True}

        # ------------------------------------------------------------------------- creation_date, birth_date, expiration_date, Id_number, and  card_number extractions -------------------------------------------------------------------
        if dict_info_frant['result'] == True:
            for i in dict_info_frant:
                if i == 'birth_date' or i == 'expiration_date':
                    dic = Date(dict_info_frant[i], i)
                    dic = dic.dates_main()
                    result[i] = dic['result']
                    if dic["decision"] == False:
                        result[
                            "Reason"] = 'Face card informations (creation_date or birth_date or expiration_date) not detected'
                        result["decision"] = False
                elif i == 'card_number':
                    dic = Numbers(dict_info_frant[i], i)
                    dic = dic.numbers_main()
                    result[i] = dic['result']
                    if dic["decision"] == False:
                        result["Reason"] = 'Face card informations (id_number or card_number) not detected'
                        result["decision"] = False
        else:
            result["decision"] = False
            result["Reason"] = 'Face card image not detected'
            remouve_images(original_frant, path_save_frant)
            return result

        result["expiration_date"] = result["expiration_date"][5:7] + '/' + result["expiration_date"][8:10] + '/' + \
                                    result["expiration_date"][0:4]
        result["birth_date"] = result["birth_date"][5:7] + '/' + result["birth_date"][8:10] + '/' + result[
                                                                                                        "birth_date"][
                                                                                                    0:4]
        remouve_images(original_frant, path_save_frant)
        return result
    except:
        return {"decision": False}


@app.post('/decode_dg_idcard')
async def decode_dg_idcard(token ,data : dict):
    return_dict={"dg2":" ","dg7":" ","dg11":" ","dg12":" "}
    dg2=data['dg2']
    dg7=data['dg7']
    dg11=data['dg11']
    dg12=data['dg12']
    try :
        decoded=codecs.getdecoder('hex_codec')(dg2)[0]
        im_start = decoded.find(b"\xFF\xD8\xFF\xE0")
        if im_start == -1:
            im_start = decoded.find(b"\x00\x00\x00\x0C\x6A\x50")
        image = decoded[im_start:]
        image = Image.open(io.BytesIO(image))
        image = np.array(image.convert("RGB"))

        image = image[:, :, ::-1]
        # cv2.imwrite(str(dg2[60:75]+'_face.png'),image)
        _, buffer = cv2.imencode('.jpg', image)  # Change the file extension as per your image format
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return_dict["dg2"]={ "result":"True",
                            "face":str(base64_image)}


    except :
            return_dict["dg2"]="False"
    try :
        decoded=codecs.getdecoder('hex_codec')(dg7)[0]
        im_start = decoded.find(b"\xFF\xD8\xFF\xE0")
        if im_start == -1:
            im_start = decoded.find(b"\x00\x00\x00\x0C\x6A\x50")
        image = decoded[im_start:]
        image = Image.open(io.BytesIO(image))
        image = np.array(image.convert("RGB"))

        image = image[:, :, ::-1]
        # cv2.imwrite(str(dg2[60:75]+'_signature.png'),image)
        _, buffer = cv2.imencode('.jpg', image)  # Change the file extension as per your image format
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return_dict[ 'dg7']={ 'result':"True",
                "signature":str(base64_image),}
    except :
            return_dict["dg7"]="False"

    try :
        if dg11:
            decoded=codecs.getdecoder('hex_codec')(dg11)[0]
            decoded=decoded.decode('iso-8859-6')
            result = ''.join([char for char in decoded if (ord(char) > 31  or  char == '&')])
            result1 = result.replace(' ', '-')
            splited_by__=result1.split('_')
            items=[word.split('<<') for word in splited_by__ ]
            surnames_latin=items[8][0]
            surnames_arabic=items[8][1][:-1]
            name_latin=items[9][0]
            name_arabic=items[9][1]
            nin=items[10][0]
            birth_date_numbers = re.findall(r'\d+', items[11][0])
            birth_date = ''.join(birth_date_numbers)
            birth_date_obj=datetime.strptime(birth_date, '%Y%m%d')
            birth_date = birth_date_obj.strftime('%Y/%m/%d')
            birth_place_latin=items[12][0]
            birth_place_latin=items[12][0]
            birth_place_arabic=items[12][1]
            blood_type=items[13][-1]
            sexe_arabic=items[13][-2]
            sexe_latin=items[13][0][-1]

            return_dict ['dg11']={'result':"True",
                    'surname_latin':surnames_latin,
                    'surname_arabic':surnames_arabic,
                    'name_latin':name_latin,
                    'name_arabic':name_arabic,
                    'birthplace_latin':birth_place_latin,
                        'birthplace_arabic':birth_place_arabic,
                    'birth_date':birth_date,
                    'sexe_latin':sexe_latin,
                    'sex_arabic':sexe_arabic,
                    'blood_type':blood_type,
                    'nin':nin }
    except :
            return_dict["dg11"]="False"
    try:
            decoded=codecs.getdecoder('hex_codec')(dg12)[0]
            decoded=decoded.decode('iso-8859-6')
            result = ''.join([char for char in decoded if ord(char) > 31 or char == '\n' or  char == '&'])
            result1 = result.replace(' ', '-')
            splited_by__=result1.split('_')
            items=[word.split('<<') for word in splited_by__ ]
            if len(items)==9:
                i=4
                deliv_date=clean_and_reform_date_format(items[i+2][0])
                exp_date=clean_and_reform_date_format(items[i+3][0])
                return_dict['dg12']={'result':"True",
                    "daira":"--",
                    "baladia_latin":items[i+1][0],
                    "baladia_arab":items[i+1][1],
                    "deliv_date":deliv_date,
                    "exp_date": exp_date}
            elif  len(items)==10:
                i=5
                deliv_date=clean_and_reform_date_format(items[i+2][0])
                exp_date=clean_and_reform_date_format(items[i+3][0])

                return_dict['dg12']={
                    'result':"True",
                    "daira":items[i][0][1:],
                    "baladia_latin":items[i+1][0],
                    "baladia_arab":items[i+1][1],
                    "deliv_date":deliv_date,
                    "exp_date": exp_date}

            else :  return_dict['dg12':'False']
    except :
        return_dict["dg12"]="False"
    return return_dict



def clean_and_reform_date_format(string_of_date):
    date_numbers = re.findall(r'\d+',string_of_date)
    date = ''.join(date_numbers)
    date_obj=datetime.strptime(date, '%Y%m%d')
    date = date_obj.strftime('%Y/%m/%d')
    return date
