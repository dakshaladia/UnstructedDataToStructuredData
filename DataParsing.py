
# imports
import os
from google.cloud import vision
import json
import io
from openai import OpenAI
import fitz
import base64
from typing import List, Dict
from pymongo import MongoClient
from pdf2image import convert_from_path
from datetime import datetime
from itertools import zip_longest
import uuid
import random
import pymongo
from pymongo import MongoClient
from datetime import datetime

poppler_path = r'poppler-24.02.0\Library\bin'

def classify_order_type_gpt4(text):
  client = OpenAI(
  api_key = "",
  )
  
  response = client.chat.completions.create(
  model="gpt-4o",
  messages= [
      {"role": "system", "content": "You are a helpful assistant proficient in classifying what order form type is the document."},
          {"role": "user", "content": f"""Classify the order type for the following document text:\n\n{text}.\nThe output should just contain the order type value in the format: 
          OrderType:<output from the classifier>.\n The order type can be one of the following:
          <List the document types>
          """
          }
      ]
  )
    
  classification = response.choices[0].message.content
  return classification

# create the prompt
def create_prompt_doctype_1(text_content):
  s = '''You are given an image of a medical document Order Form. Your task is to extract the values of the checkboxes and related text fields accurately. 
  
  Here is the some of the information you need to extract:

 <The information you need>
    
  The above is not an exhaustive list, and there can be other details present too. The document also includes handwritten annotations and selections; please be sure to capture these accurately. Analyze the provided image and return the data in a structured json format. 
  The image can contain other details not specified above too. The correct value is present to the right of the marked checkbox for parameters that have checkboxes.
  The above is just an example with dummy values, not the absolute truth. Some of the checkboxes and tabular data filled above might or might not be filled in the actual image. 

  I am describing the document format for the needed output data below. Output all the information according to the format below in JSON. All values are not neccessarily present in the document, feel free to leave values that are not present as blanks. 
  The schema in each document is parameter name followed by data type of the parameter. The parameter name is written in camel case.

  <passing the document format>

  
  Below provided is the text extracted from the same image using google vision api:

  '''
  return s + "\n" + text_content

# create the prompt
def create_prompt_doctype_2(text_content):
  s = '''You are given an image of a medical document Order Form. Your task is to extract the values of the checkboxes and related text fields accurately. 
  
  Here is the some of the information you need to extract:

 <The information you need>
    
  The above is not an exhaustive list, and there can be other details present too. The document also includes handwritten annotations and selections; please be sure to capture these accurately. Analyze the provided image and return the data in a structured json format. 
  The image can contain other details not specified above too. The correct value is present to the right of the marked checkbox for parameters that have checkboxes.
  The above is just an example with dummy values, not the absolute truth. Some of the checkboxes and tabular data filled above might or might not be filled in the actual image. 

  I am describing the document format for the needed output data below. Output all the information according to the format below in JSON. All values are not neccessarily present in the document, feel free to leave values that are not present as blanks. 
  The schema in each document is parameter name followed by data type of the parameter. The parameter name is written in camel case.

  <passing the document format>

  
  Below provided is the text extracted from the same image using google vision api:

  '''
  return s + "\n" + text_content

# Function to encode the image
def encode_image(image_content):
  # with open(image_path, "rb") as image_file:
  return base64.b64encode(image_content).decode('utf-8')

#directory check
def ensure_directory_exists(directory_path):
  if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f'Directory {directory_path} created.')
  else:
      print(f'Directory {directory_path} already exists.')

# call the prompt
def convert_to_structured_doc(text_content, page_number, image_content, order_type):    
  if("doctype_1" in order_type):
      topass = create_prompt_doctype_1(text_content)
  elif("doctype_2" in order_type):
      topass = create_prompt_doctype_2(text_content)

  # Getting the base64 string
  base64_image = encode_image(image_content)

  client = OpenAI(
  api_key = "",
  )
  
  response = client.chat.completions.create(
  model="gpt-4o",
  response_format={ "type": "json_object" },
  messages= [
    {
    "role": "user",
    "content": [
        {
        "type": "text",
        "text": f"{topass}"
        },
        {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            # "detail": "high"
        }
        }
    ]
    }
  ],  
  )

  json_output = json.dumps(response.choices[0].message.content, indent=4)
   
  return json_output

# google vision api call 
# extract content from image
# Set up the Google Vision API client
def initialize_vision_client():
  # Make sure the GOOGLE_APPLICATION_CREDENTIALS environment variable is set
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'creds.json'
  client = vision.ImageAnnotatorClient()
  return client

# Function to extract text from the Google Vision API response
def extract_text_from_response(response):
  if response.error.message:
      raise Exception(f'Error: {response.error.message}')

  text_annotations = response.text_annotations
  if not text_annotations:
      return ""

  # The first text_annotation is the full text
  full_text = text_annotations[0].description
  return full_text

# Sample function to perform text detection on an image
def detect_text(content):
  client = initialize_vision_client()
  image = vision.Image(content=content)
  response = client.text_detection(image=image)
  
  extracted_text = extract_text_from_response(response)

  return extracted_text


def convert_string_to_object(escaped_string):
  # Remove unnecessary escaped sequences
  json_string = escaped_string.replace('\\n', '').replace('\\"', '"').replace('\\\"','\"').replace('•', '-')
  json_string = json_string.strip('"')
  # Parse the JSON string into a Python object
  obj = json.loads(json_string)
  return obj
    
def read_json_files(directory: str) -> List[Dict]:
  """Read all JSON files from the given directory and return a list of JSON objects."""
  json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
  json_list = []
  for json_file in json_files:
      with open(os.path.join(directory, json_file), 'r') as file:
          json_list.append(json.load(file))
  return json_list

def merge_lists(list1: List, list2: List) -> List:
  """Merge two lists of dictionaries, preferring non-null and non-empty values for each key."""
  merged_list = []
  for item1, item2 in zip_longest(list1, list2):
      if item1 is None:
          merged_list.append(item2)
      elif item2 is None:
          merged_list.append(item1)
      elif isinstance(item1, dict) and isinstance(item2, dict):
          merged_list.append(merge_jsons([item1, item2]))
      else:
          merged_list.append(item1 if item1 not in [None, "", []] else item2)
  return merged_list

def merge_jsons(json_list: List[str]) -> Dict:
  """Merge a list of JSON objects, preferring non-null values for each key."""
  merged_json = {}
  json_obj = {}
  count =0 
  for json_str in json_list:
      if isinstance(json_str, str):
          try:
              json_obj = convert_string_to_object(json_str)
          except json.JSONDecodeError as e:
              # output_file = f'errorJson_{count}.json'
              # write_json_to_file(merged_json, output_file)
              print(f"Error decoding JSON: {e}")                
      else:
          json_obj = json_str 
          
      for key, value in json_obj.items():
          if key not in merged_json or (merged_json[key] in [None, "", []] and value not in [None, "", []]):
              merged_json[key] = value
          elif isinstance(merged_json[key], dict) and isinstance(value, dict):
              merged_json[key] = merge_jsons([merged_json[key], value])
          elif isinstance(merged_json[key], list) and isinstance(value, list):
              merged_json[key] = merge_lists(merged_json[key], value)
  return merged_json

def write_json_to_file(json_obj: Dict, filepath: str):
  """Write the given JSON object to a file."""
  with open(filepath, 'w') as file:
      json.dump(json_obj, file, indent=4)
        
class JSONEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, ObjectId):
      return str(o)
    return super(JSONEncoder, self).default(o)
        
def write_json_to_file_encoded(json_obj: Dict, filepath: str):
  """Write the given JSON object to a file."""
  with open(filepath, 'w') as file:
    json.dump(json_obj, file, cls=JSONEncoder, indent=4)

# convert pdf to image
def convert_pdf_to_image(pdf_path):  
  images = convert_from_path(pdf_path, poppler_path=poppler_path)
  return images

def convert_image_to_text(images):
  dpi = (300, 300)
  string_list=[]
  for page_number,image in enumerate(images):
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG', dpi=dpi)
    image_content = byte_io.getvalue()
    print('pix generated')
    
    text_content = detect_text(image_content)
    print('text_content generated')
    string_list.append(text_content)

  return string_list
        
def doc_ai_classifier_content(string_list):
  count =0
  full_text = ""
  for image_content in string_list:
    if(count<2):
      full_text = full_text + image_content + "\n"
    else:
      break
  return full_text

def convert_textandimage_to_json(string_list,images, order_type):
  json_list = []
  dpi = (300, 300)
  for page_number,image in enumerate(images):
    # print(page_number)
    text_content = string_list[page_number]
      
    byte_io = io.BytesIO()
    image.save(byte_io, 'PNG', dpi=dpi)
    image_content = byte_io.getvalue()
      
    json_output = convert_to_structured_doc(text_content, page_number, image_content, order_type)
              
    json_list.append(json_output)
  
  return json_list

def generate_ids_doctype_1(json_data):
  # Generate unique IDs
  json_data["patients"]["_id"] = generate_unique_id()
  
  # generate the rest of the unique ids for the tables as required.

  return json_data

def generate_ids_doctype_2(json_data):
  # Step 1: Generate a unique patient_id
  patient_id = generate_unique_id()
  json_data["patients"]["_id"] = patient_id
  
  # Step 2: Generate unique ids for other documents and update references
  # generate the rest of the unique ids for the tables as required.
  
  return json_data

def generate_unique_id():
  # return str(uuid.uuid4())
  return random.randint(1, 9223372036854775807)

def generate_ids(json_data):
  if("doctype_1" in order_type.lower()):
    json_data = generate_ids_doctype_1(json_data)
  if("doctype_2" in order_type.lower()):
    json_data = generate_ids_doctype_2(json_data)

  return json_data
    
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        return file.read()

def insert_data_into_mongodb(json_data, db_name, pdf_path, order_type):
  # Connect to MongoDB
  client = MongoClient('')
                        
  db = client[db_name] 
  
  patient_id = json_data["patients"]["_id"]
  patient_name = json_data["patients"]["firstName"] + json_data["patients"]["lastName"]
  current_datetime = datetime.now()
  dob = json_data['patients']['DOB']
  pdf_data = read_pdf(pdf_path)
  
  primary_key_data = {
      'patient_id': patient_id,
      'name': patient_name,
      'dob': dob,
      'order_type':order_type,
      'time_of_entry':current_datetime,
      'pdf':pdf_data
  }
  primary_patient_data_collection = db['primary_patient_data']
  primary_patient_data_collection.insert_one(primary_key_data)

  if("doctype_1" in order_type):    
      # Insert data into MongoDB collections
      db.patients.insert_one(json_data["patients"])
      # enter the rest of the tables in the db as per the required schema

  elif("doctype_2" in order_type):
      # Insert into patients collection
      db.patients.insert_one(json_data["patients"])
      # enter the rest of the tables in the db as per the required schema
  

  print("Data inserted successfully!")


import os
import time
from PyPDF2.errors import PdfReadError
from PyPDF2 import PdfFileReader

def is_pdf(file_path):
    try:
        with open(file_path, "rb") as file:
            PdfFileReader(file)
        return True
    except (PdfReadError, Exception):
        return False

def get_pdf_files(folder_path):
    pdf_files = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            pdf_files.append(file_path)
    return pdf_files

def process_file(file_path):
    
    pdf_path = 'fileToBeProcessed.pdf'
    images = convert_pdf_to_image(pdf_path)
    string_list = convert_image_to_text(images)
    full_text = doc_ai_classifier_content(string_list)
    order_type = classify_order_type_gpt4(full_text)
    order_type = order_type.split(':')[1].replace(" ","").lower()
   
    json_list = convert_textandimage_to_json(string_list,images, order_type)
    merged_json = merge_jsons(json_list)
    
    flag = True
    while(flag):
        try:
            outfile = generate_ids(merged_json)
        except: 
           json_list = convert_textandimage_to_json(string_list,images, order_type)
           merged_json = merge_jsons(json_list)
           outfile = generate_ids(merged_json)
        finally:
           flag = False       

    db_name = "test_db"
    insert_data_into_mongodb(outfile, db_name, pdf_path,order_type)
    print('Code executed successfully!')

def poll_folder(folder_path, delay=5):
    processed_files = set()
    
    while True:
        pdf_files = get_pdf_files(folder_path)
        for file_path in pdf_files:
            if file_path not in processed_files:
                process_file(file_path)
                processed_files.add(file_path)
        
        time.sleep(delay)

if __name__ == "__main__":
    folder_to_watch = "downloaded-files"
    poll_folder(folder_to_watch, delay=5)  # Adjust delay as necessary

