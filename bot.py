import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from image_to_image import StableDiffusionImg2ImgPipeline, preprocess
from PIL import Image

import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CallbackQueryHandler, ContextTypes, MessageHandler, CommandHandler, filters
from io import BytesIO
import random

load_dotenv()
TG_TOKEN = os.getenv('TG_TOKEN')
MODEL_DATA = os.getenv('MODEL_DATA', 'CompVis/stable-diffusion-v1-4')
LOW_VRAM_MODE = (os.getenv('LOW_VRAM', 'true').lower() == 'true')
USE_AUTH_TOKEN = os.getenv('USE_AUTH_TOKEN')
SAFETY_CHECKER = os.getenv('SAFETY_CHECKER', True)
HEIGHT = int(os.getenv('HEIGHT', '512'))
WIDTH = int(os.getenv('WIDTH', '512'))
NUM_INFERENCE_STEPS = int(os.getenv('NUM_INFERENCE_STEPS', '100'))
STRENTH = float(os.getenv('STRENTH', '0.75'))
GUIDANCE_SCALE = float(os.getenv('GUIDANCE_SCALE', '7.5'))
NUMBER_IMAGES = int(os.getenv('NUMBER_IMAGES', '1'))

revision = "fp16" if LOW_VRAM_MODE else None
torch_dtype = torch.float16 if LOW_VRAM_MODE else None

#user variables
OPTIONS_U = {}

# load the text2img pipeline
pipe = StableDiffusionPipeline.from_pretrained(MODEL_DATA, revision=revision, torch_dtype=torch_dtype, use_auth_token=USE_AUTH_TOKEN)
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
    keyboard = [[InlineKeyboardButton("Try again", callback_data="TRYAGAIN"), InlineKeyboardButton("Variations", callback_data="VARIATIONS")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    return reply_markup


def generate_image(prompt, seed=None, height=HEIGHT, width=WIDTH, num_inference_steps=NUM_INFERENCE_STEPS, strength=STRENTH, guidance_scale=GUIDANCE_SCALE, number_images=NUMBER_IMAGES, user_id=None, photo=None):
    seed = seed if seed is not None else random.randint(1, 10000)
    generator = torch.cuda.manual_seed_all(seed)
    
    u_strength = OPTIONS_U.get(user_id).get('STRENTH')
    u_guidance_scale = OPTIONS_U.get(user_id).get('GUIDANCE_SCALE')
    u_num_inference_steps = OPTIONS_U.get(user_id).get('NUM_INFERENCE_STEPS')
    
    u_strength = u_strength if isFloat(u_strength) and u_strength >= 0 and u_strength <= 1 else strength
    u_guidance_scale = u_guidance_scale if isFloat(u_guidance_scale) and u_guidance_scale >= 1 and u_strength <= 8 else guidance_scale
    u_num_inference_steps = u_num_inference_steps if isInt(u_num_inference_steps) and u_num_inference_steps >= 50 and u_num_inference_steps <= 150 else num_inference_steps
    
    if photo is not None:
        pipe.to("cpu")
        img2imgPipe.to("cuda")
        init_image = Image.open(BytesIO(photo)).convert("RGB")
        init_image = init_image.resize((height, width))
        init_image = preprocess(init_image)
        with autocast("cuda"):
            images = img2imgPipe(prompt=[prompt] * number_images, init_image=init_image,
                                    generator=generator if number_images == 1 else None,
                                    strength=u_strength,
                                    guidance_scale=u_guidance_scale,
                                    num_inference_steps=u_num_inference_steps)["sample"][0]
    else:
        pipe.to("cuda")
        img2imgPipe.to("cpu")
        with autocast("cuda"):
            images = pipe(prompt=[prompt] * number_images,
                                    generator=generator if number_images == 1 else None,
                                    strength=u_strength,
                                    height=height,
                                    width=width,
                                    guidance_scale=u_guidance_scale,
                                    num_inference_steps=u_num_inference_steps)["sample"][0]
            
    images = [images] if type(images) != type([]) else images
    seed = [seed] if type(seed) != type([]) else seed
    return images, seed


async def generate_and_send_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = u_number_images if isInt(u_number_images) and u_number_images <= 4 and u_number_images > 0 else NUMBER_IMAGES
    
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
    progress_msg = update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    im, seed = await generate_image(prompt=' '.join(context.args[1:]), seed=context.args[0], number_images=1, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{" ".join(context.args[1:])}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def generate_and_send_photo_from_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    if update.message.caption is None:
        await update.message.reply_text("The photo must contain a text in the caption", reply_to_message_id=update.message.message_id)
        return
    
    u_number_images = OPTIONS_U.get(update.message.from_user['id']).get('NUMBER_IMAGES')
    u_number_images = u_number_images if isInt(u_number_images) and u_number_images <= 4 and u_number_images > 0 else NUMBER_IMAGES
    
    progress_msg = await update.message.reply_text("Generating image...", reply_to_message_id=update.message.message_id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    im, seed = generate_image(prompt=update.message.caption, photo=photo, user_id=update.message.from_user['id'])
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    for key, value in enumerate(im):
        await context.bot.send_photo(update.effective_user.id, image_to_bytes(value), caption=f'"{update.message.caption}" (Seed: {seed[key]})', reply_markup=get_try_again_markup(), reply_to_message_id=update.message.message_id)

async def anyCommands(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if OPTIONS_U.get(update.message.from_user['id']) == None:
       OPTIONS_U[update.message.from_user['id']] = {}
    command = {
        "steps" : "NUM_INFERENCE_STEPS", 
        "strength" : "STRENTH", 
        "guidance_scale" : "GUIDANCE_SCALE", 
        "number" : "NUMBER_IMAGES"}[context.args[0]]
    if len(context.args) < 1:
        result = OPTIONS_U.get(update.message.from_user['id']).get(command)
        if result == none:
            await update.message.reply_text("had not been set", reply_to_message_id=update.message.message_id)
        else:
            await update.message.reply_text(result, reply_to_message_id=update.message.message_id)
    else:
        OPTIONS_U[update.message.from_user['id']][command] = context.args[0]
        await update.message.reply_text(f'successfully updated {command} value to {context.args[0]} ', reply_to_message_id=update.message.message_id)
    return
            
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    replied_message = query.message.reply_to_message

    await query.answer()
    progress_msg = await query.message.reply_text("Generating image...", reply_to_message_id=replied_message.message_id)

    if query.data == "TRYAGAIN":
        if replied_message.photo is not None and len(replied_message.photo) > 0 and replied_message.caption is not None:
            photo_file = await replied_message.photo[-1].get_file()
            photo = await photo_file.download_as_bytearray()
            prompt = replied_message.caption
            im, seed = generate_image(prompt, photo=photo, number_images=1)
        else:
            prompt = replied_message.text
            im, seed = generate_image(prompt, number_images=1)
    elif query.data == "VARIATIONS":
        photo_file = await query.message.photo[-1].get_file()
        photo = await photo_file.download_as_bytearray()
        prompt = replied_message.text if replied_message.text is not None else replied_message.caption
        im, seed = generate_image(prompt, photo=photo, number_images=1)
    await context.bot.delete_message(chat_id=progress_msg.chat_id, message_id=progress_msg.message_id)
    await context.bot.send_photo(update.effective_user.id, image_to_bytes(im), caption=f'"{prompt}" (Seed: {seed})', reply_markup=get_try_again_markup(), reply_to_message_id=replied_message.message_id)



app = ApplicationBuilder().token(TG_TOKEN).build()

app.add_handler(CommandHandler(["steps", "strength", "guidance_scale", "number"], anyCommands))

app.add_handler(CommandHandler("seed", generate_and_send_photo_from_seed))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, generate_and_send_photo))
app.add_handler(MessageHandler(filters.PHOTO, generate_and_send_photo_from_photo))
app.add_handler(CallbackQueryHandler(button))

app.run_polling()
