import re
import random
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, error
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Load and preprocess WhatsApp chat data
def load_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chat_data = []
    for line in lines:
        match = re.match(r"\d{2}/\d{2}/\d{2}, \d{1,2}:\d{2}\s?[ap]m - (.+?): (.+)", line)
        if match:
            sender, message = match.groups()
            chat_data.append((sender, message))
    return chat_data

# Process multiple chat files
def prepare_chatbot_data(file_paths):
    conversations = []
    messages = []  
    context_window = 3  
    for file_path in file_paths:
        chat_data = load_chat(file_path)
        messages.extend([msg for _, msg in chat_data])  
        for i in range(len(chat_data) - context_window):
            context = " ".join([chat_data[j][1] for j in range(i, i + context_window)])
            response = chat_data[i + context_window][1]
            if chat_data[i + context_window][0].lower() != "pranesh":  
                conversations.append((context, response))
    return pd.DataFrame(conversations, columns=['input', 'response']), messages

# Train chatbot
def train_chatbot(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  
    vectors = vectorizer.fit_transform(data['input'])
    return vectorizer, vectors, data

# Get chatbot response
def get_response(user_input, vectorizer, vectors, data):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors).flatten()
    best_matches = np.argsort(similarities)[-5:][::-1]  
    best_responses = [data.iloc[idx]['response'] for idx in best_matches if similarities[idx] > 0.3]
    return random.choice(best_responses) if best_responses else "Sorry, I don't understand."

# Get IST time safely
def get_ist_time():
    now = datetime.now(timezone.utc)  
    return now.astimezone(timezone(timedelta(hours=5, minutes=30)))  

# Send a message at a specific IST time
async def send_scheduled_message(application, user_id, message, target_hour, target_minute):
    ist_now = get_ist_time()
    target_time = ist_now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)  
    
    if ist_now > target_time:  
        target_time += timedelta(days=1)

    wait_time = (target_time - ist_now).total_seconds()
    await asyncio.sleep(wait_time)  
    try:
        await application.bot.send_message(chat_id=user_id, text=message)
    except error.BadRequest as e:
        print(f"Error: {e}")

# Send a random message at a random IST time
async def send_random_message(application, user_id, messages):
    ist_now = get_ist_time()
    random_hour = random.randint(10, 18)  
    random_minute = random.randint(0, 59)
    target_time = ist_now.replace(hour=random_hour, minute=random_minute, second=0, microsecond=0)  
    if ist_now > target_time:
        target_time += timedelta(days=1)
    wait_time = (target_time - ist_now).total_seconds()
    await asyncio.sleep(wait_time)
    
    random_message = random.choice(messages)
    try:
        await application.bot.send_message(chat_id=user_id, text=random_message)
    except error.BadRequest as e:
        print(f"Error: {e}")

# Initialize chatbot data
file_paths = [
    "./WhatsApp Chat with 🦋 حب(1).txt",
    "./WhatsApp Chat with 🦋 حب(2).txt",
    "./WhatsApp Chat with 🦋 حب(3).txt",
    "./WhatsApp Chat with 🦋 حب.txt"
]
data, chat_messages = prepare_chatbot_data(file_paths)
vectorizer, vectors, chatbot_data = train_chatbot(data)

TOKEN = "7874371911:AAE-H9SFpu0dILwoad_kWu3103T9JqxnfaA"
USER_ID = 5142359126  

async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! I'm your chatbot. Type a message to chat with me.")

async def chat(update: Update, context: CallbackContext):
    user_input = update.message.text
    response = get_response(user_input, vectorizer, vectors, chatbot_data)
    await update.message.reply_text(response)

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Schedule messages at the correct IST time
    loop.create_task(send_scheduled_message(app, USER_ID, "Saptiya? 😘💖", 8, 30))  # Morning 8:30 AM IST
    loop.create_task(send_scheduled_message(app, USER_ID, "Saptiya? 🍽️💖", 13, 45))  # Afternoon 1:45 PM IST
    loop.create_task(send_scheduled_message(app, USER_ID, "Saptiya? 🌙💖", 21, 30))  # Night 9:30 PM IST
    loop.create_task(send_scheduled_message(app, USER_ID, "Good morning Domar! ☀️💖", 6, 30))  # Morning 6:30 AM IST
    
    loop.create_task(send_random_message(app, USER_ID, chat_messages))  

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
