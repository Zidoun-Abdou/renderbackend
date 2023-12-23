from typing import Optional

from readmrz import MrzDetector, MrzReader
from pathlib import Path
import pytesseract
from fastapi import FastAPI,UploadFile, File,Request
import shutil
import cv2
import base64
import codecs
import re
import numpy as np
from PIL import Image
import io
from datetime import datetime


app = FastAPI()


tesseract_path = '/usr/bin/tesseract'
pytesseract = pytesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path

detector = MrzDetector()
reader = MrzReader()

@app.get("/")
async def root():
    return {"message": "Hello Worldss"}

@app.post("/mrz")
async def root(myphoto: UploadFile = File(...)):
    #try:
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
        return dict_recult
    #except Exception as e:
     #   return e






@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

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
