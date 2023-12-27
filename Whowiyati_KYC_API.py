from fastapi import FastAPI, UploadFile, File
import cv2
# import numpy as np
import shutil
import random
# from datetime import datetime
app = FastAPI()
# Initialize your models and libraries here
class LibraryLoader:
    _loaded = False
    def load_libraries(self):
        if not LibraryLoader._loaded:
            #load libraries
            import easyocr,torch
            # from passporteye import read_mrz
            # from ArabicOcr import arabicocr
            from datetime import datetime
            import re,os,pytesseract
            from pathlib import Path
            # from passporteye import read_mrz as passport_mrz_reader
            
            updated_string=str(Path.cwd())
            updated_string=updated_string.replace("\\", "/")
            self.tesseract_path='/usr/bin/tesseract'
            self.re=re
            self.os=os
            self.pytesseract=pytesseract
            self.datetime=datetime
            self.tessetact_congfig="tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ########################################## Path of card detection model and weights ###########################################################
            path_model_card_detection=updated_string+'/cardp/yolov5'
            path_model_card_detection_weights=updated_string+'/cardp/yolov5/runs/train/exp3the_bestfinal/weights/best.pt'
            ########################################## Path of card information detection model and weights ###########################################################
            path_model_info_card=updated_string+'/card_info/yolov5'
            path_model_info_card_weights=updated_string+'/card_info/yolov5/runs/train/exp47bestv1/weights/best.pt'
            #load models for just one time
            self.model_card_detection = torch.hub.load(path_model_card_detection, 'custom',path_model_card_detection_weights, source='local')
            self.model_card_detection.conf= 0.92
            self.model_info_extraction = torch.hub.load(path_model_info_card, 'custom',path_model_info_card_weights, source='local')
            #load Static variabels
            self.path_frant=updated_string+'/images/Front_card.jpg'
            self.path_back=updated_string+'/images/Back_card.jpg'
            #--------------------------------------------- path for saved croped card image ---------------------------------------------------------------
            self.path_save_frant = updated_string+'/saved_images_card_detection/image_resultR.png'
            self.path_save_back = updated_string+'/saved_images_card_detection/image_resultV.png'
            self.reader_easy_ocr = easyocr.Reader(['en'])
            self.nums = ['1','2','3','4','5','6','7','8','9','0']
            
            _loaded = True
            print('_loaded:  ',_loaded)

library_loader = LibraryLoader()
library_loader.load_libraries()
def face_card_detection(path,path_save_frant):
    dict_result={"result":True}
    try:
        results = library_loader.model_card_detection(path)
        crops = results.crop(save=False)
        image=crops[0]['im']
        name= crops[0]['label']
        name,x= name.split(' ')
        if name == 'face_card' and len(image)>0:
            dict_result.update(face_card=image)
            cv2.imwrite(path_save_frant,image)
            return dict_result
        else: 
            dict_result["result"]=False
            return dict_result
    except:
        dict_result["result"]=False
        return dict_result

#-------------------------------------------------------------------------- Frant_card info exctraction function -------------------------------------------------------------------
def info_extraction_frant(path):
    result_dic={"result": False}
    list_result=[ 'arabic_name',
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
      labels,precision,cord_point = results.xyxyn[0][:, -1], results.xyxyn[0][:,-2],results.xyxyn[0][:,:-2]
      cpt=0
      for i in labels:
          crops = results.crop(save=False)          
          crops[cpt]['label'],x= crops[cpt]['label'].split(' ')
          if crops[cpt]['label'] in list_result :
            if crops[cpt]['label'] not in result_dic.keys():
              result_dic['result']=True
              result_dic[crops[cpt]['label']]=crops[cpt]['im']
          cpt=cpt+1
      return result_dic
    except: return result_dic

    
#-------------------------------------------------------------------------------------- Card Frant detection and extraction --------------------------------------------------------------------------------------
def card_frant(path,path_save_frant):
    try:
        result_card_frant = face_card_detection(path,path_save_frant)
        if len(list(result_card_frant.keys()))==2 and list(result_card_frant.keys())[1] == 'face_card':
            detection_result = info_extraction_frant(path_save_frant)
            return detection_result
        else:
            detection_result={"result":False}
            return detection_result
    except:
        detection_result={"result":False}
        return detection_result


#------------------------------------------------------------------------------------- OCR Functions --------------------------------------------------------------------------------------------------

#####################################  dates functions   ###################################################
class Date():
    def __init__(self,image,class_name) -> None:
        self.image=image
        self.class_name=class_name
    def easy_ocr_func_dates(self):
        try:
            easy_result=library_loader.reader_easy_ocr.readtext_batched(self.image,paragraph=True)[0][-1][-1]
            if  easy_result[-1] not in library_loader.nums :
                easy_result=easy_result[:-1]
            easy_result=easy_result.replace(".", "-")
            return easy_result
        except:return 'nothing_easyocr'

    def years_between_dates(self,date1, date2):
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
    def __init__(self,image,class_name) -> None:
        self.image=image
        self.class_name=class_name
    def easy_ocr_func_number(self):
        try:
            easy_result = library_loader.reader_easy_ocr.readtext(self.image)
            conf=easy_result[0][-1]
            easy_result=easy_result[0][-2]

            if easy_result[-1] not in library_loader.nums:
                easy_result=easy_result[:-1]
            return easy_result
        except:return 'nothing_easyocr'
    def numbers_main(self) -> dict:
        if self.class_name not in ['id_number', 'card_number']:
            return {'decision': False, 'result': 'bad_class_name'}
        easy_ocr_nums = self.easy_ocr_func_number()
        length =(len(easy_ocr_nums))
        if length not in [9,18]: 
            return {'decision': False, 'result': 'bad_image'}
        if length == 9 and  self.class_name == 'card_number':
            return {'decision':  self.class_name, 'result': easy_ocr_nums}
        elif length == 18 and  self.class_name == 'id_number':
            return {'decision':  self.class_name, 'result': easy_ocr_nums}
        else:
            return {'decision': False, 'result': 'BAD_CLASS_NAME'}

#-------------------------------------------------------------------------------------------- Remouve images function ---------------------------------------------------------------------------------------------------------
def remouve_images(original_frant,path_save_frant):
    try: library_loader.os.remove("arabe.jpg")
    except: pass
    try: library_loader.os.remove("arab_dates.png")
    except: pass
    try: library_loader.os.remove('arab_number.jpg')
    except: pass
    try: library_loader.os.remove('arabe_numbers.png')
    except: pass
    try: library_loader.os.remove(original_frant)
    except: pass
    try: library_loader.os.remove(path_save_frant)
    except: pass


#----------------------------------------------------------------------------- save image cards Frant and back to start the predicttions ----------------------------------------------------------------------------------------
def save_image_card(original_frant,original_back,Frant_image_card,Back_image_card):
    cv2.imwrite(original_frant,Frant_image_card)
    cv2.imwrite(original_back,Back_image_card)

#-------------------------------------------------------------------------------------------- API Whowiyati ----------------------------------------------------------------------------------------
@app.post("/Whowiyati_KYC/")
async def Whowiyati_KYC(front_image: UploadFile = File(...)):
     try:
        path_save_frant= library_loader.path_save_frant
        original_frant= library_loader.path_frant
        with open(original_frant, "wb") as f:
                shutil.copyfileobj(front_image.file, f)
        random_float = random.uniform(0, 1)
        random_float=str(random_float)
        dict_info_frant=card_frant(original_frant,path_save_frant)
        result={'decision':True}
        
    #------------------------------------------------------------------------- creation_date, birth_date, expiration_date, Id_number, and  card_number extractions -------------------------------------------------------------------
        if dict_info_frant['result']==True:
            for i in dict_info_frant:
                    if  i == 'birth_date' or i == 'expiration_date':
                        dic=Date(dict_info_frant[i],i)
                        dic=dic.dates_main()
                        result[i]=dic['result']
                        if dic["decision"]==False:
                            result["Reason"]='Face card informations (creation_date or birth_date or expiration_date) not detected'
                            result["decision"]=False
                    elif i == 'card_number':
                        dic=Numbers(dict_info_frant[i],i)
                        dic=dic.numbers_main()
                        result[i]=dic['result']
                        if dic["decision"]==False:
                            result["Reason"]='Face card informations (id_number or card_number) not detected'
                            result["decision"]=False 
        else:
            result["decision"]=False
            result["Reason"]='Face card image not detected'
            remouve_images(original_frant,path_save_frant)
            return result  

        result["expiration_date"]=result["expiration_date"][5:7]+'/'+result["expiration_date"][8:10]+'/'+result["expiration_date"][0:4]
        result["birth_date"]=result["birth_date"][5:7]+'/'+result["birth_date"][8:10]+'/'+result["birth_date"][0:4]
        remouve_images(original_frant,path_save_frant)
        return result
     except:
         return {"decision":False}
