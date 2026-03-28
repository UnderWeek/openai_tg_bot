import asyncio
import base64
import json
import logging
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import openai
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

CONFIG_PATH = Path(__file__).with_name("config.json")
MAX_CONTEXT_PAIRS = 15
LOG_PATH = Path("bot.log")
MAX_IMAGE_B64_LEN = 45_000


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found, please copy the template and fill credentials")
    with CONFIG_PATH.open() as handler:
        return json.load(handler)


config = load_config()
AUTHORIZED_USER_ID = int(config.get("user_id", 0))
DEFAULT_MODEL = config.get("default_model")
openai.api_key = config.get("openai_api_key")


class SessionData:
    def __init__(self, internet_access: bool):
        self.history: Deque[Dict[str, str]] = deque(maxlen=MAX_CONTEXT_PAIRS)
        self.internet_access = internet_access


sessions: Dict[int, SessionData] = {}


def get_session(user_id: int) -> SessionData:
    if user_id not in sessions:
        sessions[user_id] = SessionData(bool(config.get("internet_access", False)))
    return sessions[user_id]


def is_authorized(user_id: int) -> bool:
    return user_id == AUTHORIZED_USER_ID


def build_keyboard(session: SessionData) -> InlineKeyboardMarkup:
    internet_label = "🌐 Internet Access" if session.internet_access else "🚫 Internet Access"
    markup = InlineKeyboardMarkup(row_width=3)
    markup.add(
        InlineKeyboardButton(text="🧹 Clear", callback_data="btn_clear"),
        InlineKeyboardButton(text="📚 Models", callback_data="btn_models"),
        InlineKeyboardButton(text=internet_label, callback_data="btn_internet"),
    )
    return markup


def build_models_message() -> str:
    lines = ["Available models (cost per 1M tokens):"]
    for model in config.get("models", []):
        input_price = model.get("input_price", 0)
        output_price = model.get("output_price", 0)
        lines.append(
            f"• {model.get('name')} – input ${input_price:.3f}, output ${output_price:.3f}"
        )
    return "\n".join(lines)


def log_request(user: types.User, prompt: str) -> None:
    logging.info("Request from %s (%s): %s", user.full_name, user.id, prompt)


def log_response(user: types.User, response: str) -> None:
    logging.info("Response to %s (%s): %s", user.full_name, user.id, response)


async def build_image_payload(message: types.Message) -> Optional[str]:
    photos = message.photo or []
    if not photos:
        return None
    photo = photos[-1]
    file_info = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file_info.file_path)
    encoded = base64.b64encode(file_bytes).decode()
    if len(encoded) > MAX_IMAGE_B64_LEN:
        encoded = encoded[:MAX_IMAGE_B64_LEN] + "...(truncated base64)..."
    descriptor = f"Image ({photo.width}x{photo.height}, {photo.file_size} bytes)"
    return f"{descriptor}\n{encoded}"


async def query_openai(session: SessionData, prompt: str, image_payload: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": config.get("system_message", "")}
    ]
    if not session.internet_access:
        messages.append(
            {
                "role": "system",
                "content": "Internet access is disabled; answer using internal knowledge only.",
            }
        )
    for pair in session.history:
        messages.append({"role": "user", "content": pair["user"]})
        messages.append({"role": "assistant", "content": pair["assistant"]})
    user_content = prompt.strip()
    if image_payload:
        if user_content:
            user_content = f"{user_content}\n\nAttached image data:\n{image_payload}"
        else:
            user_content = f"Attached image data:\n{image_payload}"
    if not user_content:
        user_content = "Please describe the attached image."
    messages.append({"role": "user", "content": user_content})

    response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1200,
    )
    choice = response.choices[0]
    return choice.message.content.strip()


async def ensure_authorized(message: types.Message) -> bool:
    if not is_authorized(message.from_user.id):
        await message.answer("Access denied. This bot is private.")
        return False
    return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)

bot = Bot(token=config["telegram_token"], parse_mode=types.ParseMode.HTML)
dp = Dispatcher(storage=MemoryStorage())


@dp.message.register(Command(commands=["start", "help"]))
async def handle_start(message: types.Message) -> None:
    if not await ensure_authorized(message):
        return
    session = get_session(message.from_user.id)
    await message.answer(
        "Hello! I am a secure assistant. Use the buttons below to control behavior.",
        reply_markup=build_keyboard(session),
    )


@dp.message.register(Command(commands=["models"]))
async def handle_models_command(message: types.Message) -> None:
    if not await ensure_authorized(message):
        return
    session = get_session(message.from_user.id)
    await message.answer(build_models_message(), reply_markup=build_keyboard(session))


@dp.message.register()
async def handle_user_message(message: types.Message) -> None:
    if not await ensure_authorized(message):
        return
    session = get_session(message.from_user.id)
    prompt_text = message.text or message.caption or ""
    image_payload = await build_image_payload(message)
    log_entry = prompt_text or "[image only]"
    if image_payload:
        log_entry = f"{log_entry} + image" if log_entry else "[image only] + image"
    log_request(message.from_user, log_entry)
    try:
        reply = await query_openai(session, prompt_text, image_payload=image_payload)
    except Exception as exc:
        logging.exception("OpenAI call failed")
        await message.answer(
            "Sorry, there was an error while contacting OpenAI. Please try again later."
        )
        return
    session.history.append({"user": prompt_text or "[image only]", "assistant": reply})
    log_response(message.from_user, reply)
    await message.answer(reply, reply_markup=build_keyboard(session))


@dp.callback_query.register(lambda c: c.data == "btn_clear")
async def handle_clear(query: types.CallbackQuery) -> None:
    if not is_authorized(query.from_user.id):
        await query.answer("Access denied.", show_alert=True)
        return
    session = get_session(query.from_user.id)
    session.history.clear()
    await query.answer("Conversation cleared.")
    await query.message.edit_reply_markup(reply_markup=build_keyboard(session))


@dp.callback_query.register(lambda c: c.data == "btn_models")
async def handle_models(query: types.CallbackQuery) -> None:
    if not is_authorized(query.from_user.id):
        await query.answer("Access denied.", show_alert=True)
        return
    session = get_session(query.from_user.id)
    await query.answer()
    await query.message.reply(build_models_message(), reply_markup=build_keyboard(session))


@dp.callback_query.register(lambda c: c.data == "btn_internet")
async def handle_internet_toggle(query: types.CallbackQuery) -> None:
    if not is_authorized(query.from_user.id):
        await query.answer("Access denied.", show_alert=True)
        return
    session = get_session(query.from_user.id)
    session.internet_access = not session.internet_access
    status = "enabled" if session.internet_access else "disabled"
    await query.answer(f"Internet access {status}.")
    await query.message.edit_reply_markup(reply_markup=build_keyboard(session))


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
