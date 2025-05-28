import asyncio
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram import Router, F
from aiogram.filters import Command


API_TOKEN = '7606282976:AAHbbHQsGHuXpX4XrI7xc8lUrJjn3pEM39E'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()


async def fetch_detect(image_path):
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        with open(image_path, "rb") as f:
            data.add_field('image', f, filename='213.jpg', content_type='image/jpeg')
            async with session.post('http://127.0.0.1:8000/api/detect/', data=data) as response:
                if response.status == 200:
                    result = await response.json(content_type=None)
                    return result
                else:
                    return "Error"

@router.message(F.photo)
async def send_joke(message: types.Message):
    if message.photo:
        # Получаем последнее изображение (самое высокое качество)
        photo = message.photo[-1].file_id
        image_path = 'recognize.jpg'
        await bot.download(photo, destination=image_path)

        license_plates = await fetch_detect(image_path)
        if license_plates is None:
            message_text = "Номер не обнаружен"
        else:
            message_text = f"Обнаружен номер {license_plates[0]}"
        await message.reply(message_text)

@dp.message(Command(commands=['start']))
async def send_welcome(message: types.Message):
    await message.reply("Привет! Чтобы получить случайную шутку о Чаке Норрисе, напиши /joke.")

async def main():
    dp.include_router(router)
    await dp.start_polling(bot, skip_updates=True)


if __name__ == '__main__':
    asyncio.run(main())
