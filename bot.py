import torch
from torch import autocast

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline, DDIMScheduler, LMSDiscreteScheduler
from PIL import Image, ImageChops


import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, CommandHandler, ConversationHandler, filters
from io import BytesIO
import random
from math import ceil

# REAL-ESRGAN need
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import sys
import cv2
import numpy as np


import json
import re

sys.path.insert(0, '../Real-ESRGAN ')

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
MODEL_DATA = os.getenv('MODEL_DATA', 'CompVis/stable-diffusion-v1-4')
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = os.getenv('USE_AUTH_TOKEN')
SAFETY_CHECKER = (os.getenv('SAFETY_CHECKER', 'true').lower() == 'true')
HEIGHT = int(os.getenv('HEIGHT', '512'))
WIDTH = int(os.getenv('WIDTH', '512'))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '25'))
STRENTH = float(os.getenv('STRENTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))
NUMBER_IMAGES = int(os.getenv('NUMBER_IMAGES', '1'))

SCHEDULER = os.getenv('SCHEDULER', 'None').lower()
LIMIT_SIZE = int(os.getenv('LIMIT_SIZE', '1024'))

MODEL_ESRGAN = str(os.getenv('MODEL_ESRGAN', 'generic')).lower()
MODEL_ESRGAN_ARRAY = {
  'face' : 'RestoreFormer.pth',
  'anime' : 'RealESRGAN_x4plus_anime_6B.pth',
  'generic' : 'RealESRGAN_x4plus.pth'
}

SERVER = str(os.getenv('SERVER', "https://api.telegram.org"))


revision = "fp16" if LOW_VRAM_MODE else None
torch_dtype = torch.float16 if LOW_VRAM_MODE else None

#user variables
OPTIONS_U = {}


OPTION_JSON_FILE = "user_variables.json"
if os.path.exists('/content/drive/MyDrive/Colab/StableDiffusionTelegram/' + OPTION_JSON_FILE) is True:
  try:
    with open('/content/drive/MyDrive/Colab/StableDiffusionTelegram/' + OPTION_JSON_FILE, 'r') as file:
      OPTIONS_U = json.load(file)
  except:
    False
   
# Text-to-Image Scheduler 
# - PLMS from StableDiffusionPipeline (Default)
# - DDIM 
# - K-LMS

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False) if SCHEDULER is "ddim" else \
            LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,  beta_schedule="scaled_linear") if SCHEDULER is "klms" else \
            None


# load the text2img pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_DATA, scheduler=scheduler, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN) if scheduler is not None else \
       StableDiffusionPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
            
pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

# load the img2img pipeline
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DATA, scheduler=scheduler, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN) if scheduler is not None else \
              StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
img2imgPipe = img2imgPipe.to("cpu")
img2imgPipe.enable_attention_slicing()

inpaint2imgPipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_DATA, scheduler=scheduler, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN) if scheduler is not None else \
                  StableDiffusionInpaintPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
inpaint2imgPipe = inpaint2imgPipe.to("cpu")
inpaint2imgPipe.enable_attention_slicing()
# disable safety checker if wanted
def dummy_checker(images, **kwargs): return images, False
if not SAFETY_CHECKER:
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker
    inpaint2imgPipe.safety_checker = dummy_checker
    
def isInt(input):
    try:
       int(input)
       return True
    except:
       return False

def isFloat(input):
    try:
       float(input)
       return True
    except:
       return False

def image_to_bytes(image):
    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)
    return bio

def get_try_again_markup():
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")],\
                [InlineKeyboardButton("Upscale", callback_data="UPSCALE4"), InlineKeyboardButton("Restore", callback_data="RESTORE")],\
                [InlineKeyboardButton("Inpaint", callback_data="INPAINT")]]

    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

def get_exit_inpaint_markup():
   keyboard = [[KeyboardButton("Exit from inpainting")]]
   reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
   return reply_markup

def restore_image(input):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    
    #restorer
    upsampler = RealESRGANer(
        scale=2,
        model_path=os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY['generic']),
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False)
        
    face_enhancer = GFPGANer(
        model_path=os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY['face']),
        upscale=1,
        arch='RestoreFormer',
        channel_multiplier=2,
        bg_upsampler=upsampler)
        
    _, _, output = face_enhancer.enhance(cv2.imdecode(np.array(input)), has_aligned=False, only_center_face=False, paste_back=True)
    return output  

def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENTH, guidance_scale=GUIDANCE_SCALE, number_images=None, user_id=None, photo=None, inpainting=None):
    seed = seed if isInt(seed) is True else random.randint(1, 1000000) if seed is None else None
    generator = torch.cuda.manual_seed_all(seed) if seed is not None else None
    
    if OPTIONS_U.get(user_id) == None:
       OPTIONS_U[user_id] = {}
    
    u_strength = OPTIONS_U.get(user_id).get('STRENTH')
    u_guidance_scale = OPTIONS_U.get(user_id).get('GUIDANCE_SCALE')
    u_num_inference_steps = OPTIONS_U.get(user_id).get('NUM_INFERENCE_STEPS')
    u_number_images = OPTIONS_U.get(user_id).get('NUMBER_IMAGES')
    u_width = OPTIONS_U.get(user_id).get('WIDTH')
    u_height = OPTIONS_U.get(user_id).get('HEIGHT')
    
    u_strength = float(u_strength) if isFloat(u_strength) and float(u_strength) >= 0 and float(u_strength) <= 1 else strength
    u_guidance_scale = float(u_guidance_scale) if isFloat(u_guidance_scale) and float(u_guidance_scale) >= 1 and float(u_strength) <= 30 else guidance_scale
    u_num_inference_steps = int(u_num_inference_steps) if isInt(u_num_inference_steps) and int(u_num_inference_steps) >= 1 and int(u_num_inference_steps) <= 150 else num_inference_steps
    u_number_images = int(u_number_images) if isInt(u_number_images) and int(u_number_images) >= 1 and int(u_number_images) <= 4 else NUMBER_IMAGES
    u_width = WIDTH if isInt(u_width) is not True else 1024 if int(u_width) > 1024 else 256 if int(u_width) < 256 else int(u_width)
    u_height = HEIGHT if isInt(u_height) is not True else 1024 if int(u_height) > 1024 else 256 if int(u_height) < 256 else int(u_height)
    
    if isInt(LIMIT_SIZE) is True and LIMIT_SIZE > 256:
      limit_size_ = LIMIT_SIZE
    else:
      limit_size_ = 1024
    
    u_width = WIDTH if isInt(u_width) is not True else limit_size_ if int(u_width) > limit_size_ else 256 if int(u_width) < 256 else int(u_width)
    u_height = HEIGHT if isInt(u_height) is not True else limit_size_ if int(u_height) > limit_size_ else 256 if int(u_height) < 256 else int(u_height)
    
    if photo is not None:
        pipe.to("cpu")

        if inpainting is not None:
          img2imgPipe.to("cpu")
          inpaint2imgPipe.to("cuda")
          
        else:
          img2imgPipe.to("cuda")
          inpaint2imgPipe.to("cpu")

        if inpainting is not None and inpainting.get('base_inpaint') is not None:
          photo_ = Image.open(BytesIO(inpainting.get('base_inpaint')))
        else:
          photo_ = Image.open(BytesIO(photo))
          
        width = photo_.width
        height = photo_.height
        
        downscale = 1 if max(height, width) <= limit_size_ else max(height, width) / limit_size_
        
        downscale = 1 if max(height, width) <= limit_size_ else max(height, width) / limit_size_
         
        u_height = ceil(height / downscale)
        u_width = ceil(width / downscale)

        with autocast("cuda"):
            if inpainting is not None and inpainting.get('base_inpaint') is not None:
              
              init_image = Image.open(BytesIO(inpainting['base_inpaint'])).convert("RGB")
              init_mask = Image.open(BytesIO(photo)).convert("RGB")
              
              # Why do I rotated it? Telegram always rotate the document image. Because, idk. 
              # So I hope this will fixed it for inpainting document image. Regular img2img.. nope.
              if (init_mask.height > init_mask.width) != (init_image.height > init_image.width):
                init_image = init_image.transpose(Image.ROTATE_270)
                u_tmp = u_width
                u_width = u_height
                u_height = u_tmp
                
              init_image = init_image.resize((u_width - (u_width % 64) , u_height - (u_height % 64) ))
              init_mask = init_mask.resize((u_width - (u_width % 64) , u_height - (u_height % 64) ))
              
              # Difference to find which pixel are different between two images, 
              # Convert(L) is to convert to grayscale
              mask_area = ImageChops.difference(init_image.convert("L"), init_mask.convert("L")) 
              mask_area = mask_area.point(lambda x : 255 if x > 10 else 0 ) #Threshold
              mask_area = mask_area.convert("1") # Convert to binary (only black and white color)
              mask_area = mask_area.resize((u_width - (u_width % 64) , u_height - (u_height % 64) ))

              images = inpaint2imgPipe(prompt=[prompt] * u_number_images,
                                    generator=generator, 
                                    init_image=init_image,
                                    mask_image=mask_area,
                                    strength=u_strength,
                                    guidance_scale=u_guidance_scale,
                                   #num_inference_steps=u_num_inference_steps).images
                                    num_inference_steps=u_num_inference_steps)["sample"]

            else:
                init_image = Image.open(BytesIO(photo)).convert("RGB")
                init_image = init_image.resize((u_width - (u_width % 64) , u_height - (u_height % 64) ))
                images = img2imgPipe(prompt=[prompt] * u_number_images, 
                                     init_image=init_image,
                                     generator=generator, 
                                     strength=u_strength,
                                     guidance_scale=u_guidance_scale,
                                    #num_inference_steps=u_num_inference_steps).images
                                     num_inference_steps=u_num_inference_steps)["sample"]
            
    else:
        pipe.to("cuda")
        inpaint2imgPipe.to("cpu")
        img2imgPipe.to("cpu")
        with autocast("cuda"):
            images = pipe(prompt=[prompt] * u_number_images,
                          generator=generator, #generator if u_number_images == 1 else None,
                          strength=u_strength,
                          height=u_height - (u_height % 64),
                          width=u_width - (u_width % 64),
                          guidance_scale=u_guidance_scale,
                         #num_inference_steps=u_num_inference_steps).images
                          num_inference_steps=u_num_inference_steps)["sample"]

            
    images = [images] if type(images) != type([]) else images
    
    # resize to original form
    images = [Image.open(image_to_bytes(output_image)).resize((u_width, u_height)) for output_image in images]
    seeds = ["Empty"] * len(images)
    seeds[0] = seed if seed is not None else "Empty"  #seed if u_number_images == 1 and seed is not None else "Empty"
     
    return images, seeds


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('late_photo') is not None:
      context.user_data['late_prompt'] = update.message.text
      await generate_and_send_photo_from_photo(update, context)
      context.user_data.clear()
      return
    
    if context.user_data.get('base_inpaint') is not None:
      await end_inpainting(update, context)
    
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = NUMBER_IMAGES if isInt(u_number_images) is not True else 1 if int(u_number_images) < 1 else 4 if int(u_number_images) > 4 else int(u_number_images)

    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=update.message.text, number_images=u_number_images, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_document(update.effective_user.id, document=image_to_bytes(value), caption=f'"{update.message.text}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
    
async def generate_and_send_photo_from_seed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.user_data.get('late_photo') is not None:
      context.user_data['late_prompt'] = update.message.text
      await generate_and_send_photo_from_photo(update, context)
      context.user_data.clear()
      return
    
    if context.user_data.get('base_inpaint') is not None:
      await end_inpainting(update, context)
    
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    
    if len(context.args) < 2:
        await update.message.reply_text("The prompt was not added", reply_to_message_id=update.message.message_id)
        return
    
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')    
    u_number_images = NUMBER_IMAGES if isInt(u_number_images) is not True else 1 if int(u_number_images) < 1 else 4 if int(u_number_images) > 4 else int(u_number_images)    
    
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=' '.join(context.args[1:]), seed=context.args[0], number_images=u_number_images, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_document(update.effective_user.id, document=image_to_bytes(value), caption=f'"{" ".join(context.args[1:])}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
       
    if update.message.caption is None and \
       context.user_data.get('wait_for_base') is not True and \
       context.user_data.get('late_prompt') is None and \
       context.user_data.get('wait_for_restore') is not True:
        if update.message.document is not None:
          context.user_data['late_photo'] = update.message.document
          await update.message.reply_text("Now please type in the prompt", reply_to_message_id=update.message.message_id)
          return
        else:
          await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
  
    prompt = update.message.caption or context.user_data.get('late_prompt') or ""
    
    command = None if prompt.split(" ")[0] not in ["/seed", "/inpaint", "/inpainting", "/restore"] else prompt.split(" ")[0]
    seed = None if command is None else prompt.split(" ")[1] if command == "/seed" else None
    prompt = prompt if command is None else " ".join(prompt.split(" ")[(2 if command == "/seed" else 1):])

    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = NUMBER_IMAGES if isInt(u_number_images) is not True else 1 if int(u_number_images) < 1 else 4 if int(u_number_images) > 4 else int(u_number_images)
    
    reply_text = "Inpainting Process..." if  (context.user_data.get('base_inpaint') is not None) is True else "Generating image..."
    
    progress_msg = await update.message.reply_text(reply_text, reply_to_message_id=update.message.message_id)
    
    if len(update.message.photo) > 0:
      photo_file = await update.message.photo[-1].get_file()
    elif context.user_data.get('late_photo'):
      photo_file = await context.user_data.get('late_photo').get_file()
    else:
      photo_file = await update.message.document.get_file()
    
    
    if "0.0.0.0" in SERVER:
      photo_ = Image.open(photo_file.file_path)
      photo = image_to_bytes(photo_).read()
    else:
      photo = await photo_file.download_as_bytearray()
    
    base_inpaint = context.user_data.get('base_inpaint')
    if context.user_data.get('wait_for_base') is True or command in ["/inpaint","/inpainting"]:
      context.user_data['base_inpaint'] = photo
      context.user_data['wait_for_base'] = False
      
      await update.message.reply_text(f'Now please put a masked image', reply_to_message_id=update.message.message_id, reply_markup=get_exit_inpaint_markup())
    elif command == "/restore" or context.user_data.get('wait_for_restore') is True: 
       photo = image_to_bytes(Image.open(restore_image(photo))).read()
       await context.bot.send_document(update.effective_user.id, document=photo, caption='', reply_to_message_id=replied_message.message_id)
       context.user_data.clear()
    else:   
     #im, seed = generate_image(prompt=prompt, seed=seed, width=width, height=height, photo=photo, user_id=update.message.from_user['id'], inpainting=context.user_data if context.user_data is not None else None)
      if base_inpaint is not None:
        context.user_data['mask_image'] = photo
      
      im, seed = generate_image(prompt=prompt, seed=seed, photo=photo, user_id=update.message.from_user['id'], inpainting=(context.user_data if base_inpaint is not None else None))
      for key, value in enumerate(im):
       #await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{update.message.caption}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
        await context.bot.send_document(update.effective_user.id, document=image_to_bytes(value), caption=f'"{update.message.caption}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    
async def anyCommands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    option = "".join((update.message.text).split(" ")[0][1:]).lower()
    
    if (option in ["inpaint","inpainting"]):
      if context.user_data.get('late_photo') is not None:
        context.user_data['late_prompt'] = "/" + option
        await generate_and_send_photo_from_photo(update, context)
      else:
        context.user_data['wait_for_base'] = True
        await update.message.reply_text("Please put the image to start inpainting", reply_to_message_id=update.message.message_id, reply_markup=get_exit_inpaint_markup())
      return
    elif option == "restore":
      context.user_data['wait_for_restore'] = True
      await update.message.reply_text("Please put the image to start restoration", reply_to_message_id=update.message.message_id)
      return
    await end_inpainting(update, context)
    
    options = {
        "steps" : 'NUM_INFERENCE_STEPS' , 
        "strength" : 'STRENTH', 
        "guidance_scale" : 'GUIDANCE_SCALE', 
        "number" : 'NUMBER_IMAGES', 
        "width" : 'WIDTH', 
        "height" : 'HEIGHT',
        "model_esrgan" : 'MODEL_ESRGAN'

    }[option]
    if options is not None:
      if OPTIONS_U.get(update.message.from_user['id']) == None:
        OPTIONS_U[update.message.from_user['id']] = {}
      if len(context.args) < 1:

        result = OPTIONS_U.get(update.message.from_user['id']).get(options)
        if result == None:
          await update.message.reply_text("had not been set", reply_to_message_id=update.message.message_id)
        else:
          await update.message.reply_text(result, reply_to_message_id=update.message.message_id)
      else:
        OPTIONS_U[update.message.from_user['id']][options] = context.args[0]
        
        json_path = '/content/drive/MyDrive/Colab/StableDiffusionTelegram'
        if os.path.exists(f"{json_path}/{OPTION_JSON_FILE}") is True:
          with open(f"{json_path}/{OPTION_JSON_FILE}", 'w') as file:
            json.dump(OPTIONS_U, file, indent = 4)
        
          
        await update.message.reply_text(f'successfully updated {options} value to {context.args[0]} ', reply_to_message_id=update.message.message_id)
      
    return
            
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message
    
    if query.data == "EXIT_INPAINT":
      end_inpainting(context)
      return
    
    prompt = replied_message.caption if replied_message.caption != None else replied_message.text 
    seed = None if prompt.split(" ")[0] != "/seed" else prompt.split(" ")[1]
    prompt = prompt if prompt.split(" ")[0] != "/seed" else " ".join(prompt.split(" ")[2:])
    
    if len(query.message.photo) > 0 or query.message.document is not None:
      if len(query.message.photo) > 0:
        photo_file = await query.message.photo[-1].get_file()
      else:
        photo_file = await query.message.document.get_file()
      photo = await photo_file.download_as_bytearray()
      
    await query.answer()
  
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=query.message.message_id)
    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            
            if context.user_data.get('mask_image') is not None:
              photo = context.user_data['mask_image']
            else:
              await end_inpainting(update, context)
              if "0.0.0.0" in SERVER:
                photo = Image.open(photo_file.file_path)
                photo = image_to_bytes(photo).read()
              else:
                photo = await photo_file.download_as_bytearray()
            
            
           #im, seed = generate_image(prompt, seed=seed, width=width, height=height, photo=photo, number_images=1, user_id=replied_message.chat.id)
            base_inpaint = context.user_data.get('base_inpaint')
            mask_image = context.user_data.get('mask_image')
            im, seed = generate_image(prompt, seed=seed, photo=photo, number_images=1, user_id=replied_message.chat.id, inpainting=(context.user_data if base_inpaint is not None and mask_image is not None else None) )
        else:
            im, seed = generate_image(prompt, seed=seed, number_images=1, user_id=replied_message.chat.id)
    elif query.data == "VARIATIONS":
       #im, seed = generate_image(prompt, seed=seed, width=width, height=height, photo=photo, number_images=1, user_id=replied_message.chat.id)
        im, seed = generate_image(prompt, seed=seed, photo=photo, number_images=1, user_id=replied_message.chat.id)

    elif query.data == "UPSCALE4":

        if OPTIONS_U.get(replied_message.chat.id) is None:
            OPTIONS_U[replied_message.chat.id] = {}
            
        u_model_esrgan = OPTIONS_U[replied_message.chat.id].get('MODEL_ESRGAN')
        u_model_esrgan = u_model_esrgan if u_model_esrgan in ['generic','face', 'anime'] else 'generic'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=2) if u_model_esrgan == 'anime' else \
                RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2) 
        
        model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY[u_model_esrgan] if u_model_esrgan is 'anime' else MODEL_ESRGAN_ARRAY['generic']) 
    
        #restorer
        upsampler = RealESRGANer(
        scale=2,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False)
        
        if u_model_esrgan == 'face':
            face_enhancer = GFPGANer(
              model_path=os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY['face']),
              upscale=2,
              arch='RestoreFormer',
              channel_multiplier=2,
              bg_upsampler=upsampler)
        
        if u_model_esrgan == 'face':
            _, _, output = face_enhancer.enhance(cv2.imdecode(np.array(photo)), has_aligned=False, only_center_face=False, paste_back=True)
        else:
          #photo
          #cv2.imdecode(np.asarray(Image.open(photo).tobytes()), -1)
          output, _ = upsampler.enhance(cv2.imdecode(np.asarray(photo), -1), outscale=2)
          
    if query.data == "UPSCALE4":
        output_width  = output.shape[0]
        output_height = output.shape[1]
        image_opened  = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_image  = BytesIO()
        image_opened.save(output_image, 'jpeg', quality=80)
        
        ################
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        await context.bot.send_document(update.effective_user.id, document=output_image.getvalue(), caption=f'"{prompt}" ( {output_width}x{output_height} | {filename})', reply_to_message_id=query.message.message_id)
    
    elif query.data == "INPAINT":
       context.user_data['base_inpaint'] = photo
       
       await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
       await query.message.reply_text(f'Now please put a masked image', reply_to_message_id=replied_message.message_id, reply_markup=get_exit_inpaint_markup())
    elif query.data == "RESTORE":
       photo = image_to_bytes(Image.open(restore_image(photo))).read()
       await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
       await context.bot.send_document(update.effective_user.id, document=photo, caption='', reply_to_message_id=replied_message.message_id)
    else:
       await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
       for key, value in enumerate(im): 
          await context.bot.send_document(update.effective_user.id, document=image_to_bytes(value), caption=f'"{prompt}" (Seed: {seed[0]})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)
          
    
async def end_inpainting(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if len(context.user_data) > 0:
      suicide = await context.bot.send_message(update.effective_user.id, "Leaving Inpainting", reply_markup=ReplyKeyboardRemove())
      context.user_data.clear()
      await context.bot.delete_message(chat_id=suicide.chat_id, message_id=suicide.message_id)
    return

    
app = ApplicationBuilder() \
 .base_url(f"{SERVER}/bot") \
 .base_file_url(f"{SERVER}/file/bot") \
 .token(TG_TOKEN).build()

app.add_handler(CommandHandler(["steps", "strength", "guidance_scale", "number", "width", "height", "model_esrgan", "inpaint", "inpainting", "restore"], anyCommands))


app.add_handler(CommandHandler("seed", generate_and_send_photo_from_seed))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.Regex('^(?!Exit from inpainting)$'), generate_and_send_photo))
app.add_handler(MessageHandler(filters.Regex('^Exit from inpainting$'),end_inpainting))
app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, generate_and_send_photo_from_photo))


app.add_handler(CallbackQueryHandler(button))
app.run_polling()
