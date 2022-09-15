import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, DDIMScheduler, LMSDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess
#from image_to_image import preprocess
#from StableDiffusionImg2ImgPipeline import preprocess
from PIL import Image

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, CommandHandler, filters
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
sys.path.insert(0, '../Real-ESRGAN ')

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
MODEL_DATA = os.getenv('MODEL_DATA', 'CompVis/stable-diffusion-v1-4')
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = os.getenv('USE_AUTH_TOKEN')
SAFETY_CHECKER = False if os.getenv('SAFETY_CHECKER', True) == 'False' else True
HEIGHT = int(os.getenv('HEIGHT', '512'))
WIDTH = int(os.getenv('WIDTH', '512'))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '100'))
STRENTH = float(os.getenv('STRENTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))
NUMBER_IMAGES = int(os.getenv('NUMBER_IMAGES', '1'))
SCHEDULER = os.getenv('SCHEDULER', None)

MODEL_ESRGAN = str(os.getenv('MODEL_ESRGAN', 'generic')).lower()
MODEL_ESRGAN_ARRAY = {
  'face' : 'GFPGANv1.4.pth',
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
    with open('/content/drive/MyDrive/Colab/StableDiffusionTelegram/' + OPTION_JSON_FILE, 'w') as file:
      OPTIONS_U = json.load(file)
  except:
    False
  


# Text-to-Image Scheduler 
# - PLMS from StableDiffusionPipeline (Default)
# - DDIM 
# - K-LMS
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False) if SCHEDULER is "DDIM" else \
            LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,  beta_schedule="scaled_linear") if SCHEDULER is "KLMS" else \
            None


# load the text2img pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_DATA, scheduler=scheduler, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN) if scheduler is not None else \
       StableDiffusionPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
            
pipe = pipe.to("cpu")

# load the img2img pipeline
img2imgPipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
img2imgPipe = img2imgPipe.to("cpu")

# disable safety checker if wanted
def dummy_checker(images, **kwargs): return images, False
if not SAFETY_CHECKER:
    pipe.safety_checker = dummy_checker
    img2imgPipe.safety_checker = dummy_checker

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
    keyboard = [[InlineKeyboardButton("Try again", callback_data={"TRYAGAIN"}), InlineKeyboardButton("Variations", callback_data="VARIATIONS")],\
                [InlineKeyboardButton("Upscaling", callback_data={"UPSCALE4"})]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup

def get_download_markup(input_path):
    
    keyboard = [[InlineKeyboardButton("Download", callback_data={"DOWNLOAD" : input_path ], location_file=input_path)]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup
    
def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENTH, guidance_scale=GUIDANCE_SCALE, number_images=None, user_id=None, photo=None):
    seed = seed if isInt(seed) is True else random.randint(1, 10000) if seed is None else None
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
    u_guidance_scale = float(u_guidance_scale) if isFloat(u_guidance_scale) and float(u_guidance_scale) >= 1 and float(u_strength) <= 16 else guidance_scale
    u_num_inference_steps = int(u_num_inference_steps) if isInt(u_num_inference_steps) and int(u_num_inference_steps) >= 50 and int(u_num_inference_steps) <= 150 else num_inference_steps
    u_number_images = int(u_number_images) if isInt(u_number_images) and int(u_number_images) >= 1 and int(u_number_images) <= 4 else NUMBER_IMAGES
    u_width = WIDTH if isInt(u_width) is not True else 1024 if int(u_width) > 1024 else 256 if int(u_width) < 256 else int(u_width)
    u_height = HEIGHT if isInt(u_height) is not True else 1024 if int(u_height) > 1024 else 256 if int(u_height) < 256 else int(u_height)
    
    if photo is not None:
        pipe.to("cpu")
        img2imgPipe.to("cuda")
        img2imgPipe.enable_attention_slicing()
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        
        downscale = 1 if max(height, width) <= 1024 else max(height, width) / 1024
        
        u_height = ceil(height / downscale)
        u_width = ceil(width / downscale)
        init_image = init_image.resize((u_width - (u_width % 8) , u_height - (u_height % 8) ))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            images = img2imgPipe(prompt=[prompt] * u_number_images, init_image=init_image,
                                    generator=generator, #generator if u_number_images == 1 else None,
                                    strength=u_strength,
                                    guidance_scale=u_guidance_scale,
                                    num_inference_steps=u_num_inference_steps)["sample"]
            
            
           
    else:
        pipe.to("cuda")
        pipe.enable_attention_slicing()
        
        img2imgPipe.to("cpu")
        with autocast("cuda"):
            images = pipe(prompt=[prompt] * u_number_images,
                                    generator=generator, #generator if u_number_images == 1 else None,
                                    strength=u_strength,
                                    height=u_height - (u_height % 64),
                                    width=u_width - (u_width % 64),
                                    guidance_scale=u_guidance_scale,
                                    num_inference_steps=u_num_inference_steps)["sample"]
            
    images = [images] if type(images) != type([]) else images
    
    # resize to original form
    images = [Image.open(image_to_bytes(output_image)).resize((u_width, u_height)) for output_image in images]
    
    seeds = ["Empty"] * len(images)
    seeds[0] = seed if seed is not None else "Empty"  #seed if u_number_images == 1 and seed is not None else "Empty"
     
    return images, seeds


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = NUMBER_IMAGES if isInt(u_number_images) is not True else 1 if int(u_number_images) < 1 else 4 if int(u_number_images) > 4 else int(u_number_images)
  
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = generate_image(prompt=update.message.text, number_images=u_number_images, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{update.message.text}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
    
async def generate_and_send_photo_from_seed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{" ".join(context.args[1:])}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
    
    width = update.message.photo[-1].width
    height = update.message.photo[-1].height
    
    prompt = update.message.caption
    seed = None if prompt.split(" ")[0] != "/seed" else prompt.split(" ")[1]
    prompt = prompt if prompt.split(" ")[0] != "/seed" else " ".join(prompt.split(" ")[2:])
            
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = NUMBER_IMAGES if isInt(u_number_images) is not True else 1 if int(u_number_images) < 1 else 4 if int(u_number_images) > 4 else int(u_number_images)
    
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    im, seed = generate_image(prompt=prompt, seed=seed, width=width, height=height, photo=photo, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{update.message.caption}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)
 
async def anyCommands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    options = {
        "steps" : 'NUM_INFERENCE_STEPS' , 
        "strength" : 'STRENTH', 
        "guidance_scale" : 'GUIDANCE_SCALE', 
        "number" : 'NUMBER_IMAGES', 
        "width" : 'WIDTH', 
        "height" : 'HEIGHT',
        "model_esrgan" : 'MODEL_ESRGAN'
    }["".join((update.message.text).split(" ")[0][1:])]
    
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
        if os.path.exists(json_path) is True:
          with open(f"{json_path}/{OPTION_JSON_FILE}", 'w') as file:
            json.dump(OPTIONS_U, file, indent = 4)
        
          
        await update.message.reply_text(f'successfully updated {options} value to {context.args[0]} ', reply_to_message_id=update.message.message_id)
    return
            
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message
    
    prompt = replied_message.caption if replied_message.caption != None else replied_message.text 
    seed = None if prompt.split(" ")[0] != "/seed" else prompt.split(" ")[1]
    prompt = prompt if prompt.split(" ")[0] != "/seed" else " ".join(prompt.split(" ")[2:])
    
    if query.message.photo is not None:
      width = query.message.photo[-1].width
      height = query.message.photo[-1].height
    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)
    if "TRYAGAIN" in query.data:
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            im, seed = generate_image(prompt, seed=seed, width=width, height=height, photo=photo, number_images=1, user_id=replied_message.chat.id)
        else:
            im, seed = generate_image(prompt, seed=seed, number_images=1, user_id=replied_message.chat.id)
    elif "VARIATIONS" in query.data:
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        im, seed = generate_image(prompt, seed=seed, width=width, height=height, photo=photo, number_images=1, user_id=replied_message.chat.id)
    elif "UPSCALE4" in query.data:
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        if OPTIONS_U.get(replied_message.chat.id) is None:
            OPTIONS_U[replied_message.chat.id] = {}
            
        u_model_esrgan = OPTIONS_U[replied_message.chat.id].get('MODEL_ESRGAN')
        u_model_esrgan = u_model_esrgan if u_model_esrgan in ['generic','face', 'anime'] else 'generic'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4) if u_model_esrgan == 'anime' else \
                RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) 
        
        model_path = os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY[u_model_esrgan] if u_model_esrgan is 'anime' else MODEL_ESRGAN_ARRAY['generic']) 
    
        #restorer
        upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=False)
        
        if u_model_esrgan == 'face':
            face_enhancer = GFPGANer(
              model_path=os.path.join('Real-ESRGAN/experiments/pretrained_models', MODEL_ESRGAN_ARRAY['face']),
              upscale=4,
              arch='clean',
              channel_multiplier=2,
              bg_upsampler=upsampler)
        
        if u_model_esrgan == 'face':
            _, _, output = face_enhancer.enhance(cv2.imdecode(np.array(photo)), has_aligned=False, only_center_face=False, paste_back=True)
        else:
          output, _ = upsampler.enhance(cv2.imdecode(np.asarray(photo), -1), outscale=4)
           
    if 'UPSCALE4' in query.data:
        output_width  = output.shape[0]
        output_height = output.shape[1]
        image_opened  = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        output_image  = BytesIO()
        image_opened.save(output_image, 'jpeg', quality=100)
        
        ################
        save_location = '/content/output_scaled'
        if os.path.exists(save_location):
          while True:
            image_saved = f'{save_location}/{ceil(random.random() * 1000000000000)}.png'
            if os.path.exists(image_saved) is not True:
              cv2.imwrite(image_saved, output)
              break
        await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
        if os.path.exists(image_saved):
          await context.bot.send_photo(update.effective_user.id, output_image.getvalue(), caption=f'"{prompt}" (Ratio Sizes : {output_width}x{output_height})', reply_markup=get_download_markup(image_saved), reply_to_message_id=replied_message.message_id)
        else:
          await context.bot.send_photo(update.effective_user.id, output_image.getvalue(), caption=f'"{prompt}" (Ratio Sizes : {output_width}x{output_height})', reply_to_message_id=replied_message.message_id)
    elif 'DOWNLOAD' in query.data:
       
       await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
       print(query)
       await context.bot.send_photo(update.effective_user.id, file=query.data['DOWNLOAD'], reply_to_message_id=replied_message.message_id)
    else:
       await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
       for key, value in enumerate(im): 
          await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{prompt}" (Seed: {seed[0]})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)
         


app = ApplicationBuilder() \
 .base_url(f"{SERVER}/bot") \
 .base_file_url(f"{SERVER}/file/bot") \
 .token(TG_TOKEN).build() \

app.add_handler(CommandHandler(["steps", "strength", "guidance_scale", "number", "width", "height"], anyCommands))

app.add_handler(CommandHandler("seed", generate_and_send_photo_from_seed))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(MessageHandler(filters.PHOTO, generate_and_send_photo_from_photo))
app.add_handler(CallbackQueryHandler(button))

app.run_polling()
