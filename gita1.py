import os
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text

    # Send query to FastAPI backend
    response = requests.post(
        "http://127.0.0.1:8000/answer",
        json={"input": user_input}
    )
    
    if response.status_code == 200:
        answer = response.json().get("answer", "Sorry, no answer found.")
    else:
        answer = "Something went wrong. Please try again."

    await update.message.reply_text(answer)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot is running...")
    app.run_polling()
