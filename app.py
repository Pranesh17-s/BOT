import re
import json
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import os

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

# Process chat files
def prepare_chatbot_data(file_paths):
    conversations = []
    context_window = 3  # Use last 3 messages as context
    for file_path in file_paths:
        chat_data = load_chat(file_path)
        for i in range(len(chat_data) - context_window):
            context = " ".join([chat_data[j][1] for j in range(i, i + context_window)])
            response = chat_data[i + context_window][1]
            if chat_data[i + context_window][0].lower() != "pranesh":  # Exclude Pranesh's messages as training data
                conversations.append((context, response))
    return pd.DataFrame(conversations, columns=['input', 'response'])

# Train chatbot
def train_chatbot(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use bigrams for better matching
    vectors = vectorizer.fit_transform(data['input'])
    return vectorizer, vectors, data

# Get chatbot response
def get_response(user_input, vectorizer, vectors, data):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, vectors).flatten()
    best_matches = np.argsort(similarities)[-3:][::-1]  # Get top 3 responses
    best_responses = [data.iloc[idx]['response'] for idx in best_matches if similarities[idx] > 0.2]
    return random.choice(best_responses) if best_responses else "Sorry, I don't understand."

# Initialize chatbot data (Use relative paths to load chat files in the same directory)
file_paths = ["./WhatsApp Chat with 🦋 حب(1).txt", "./WhatsApp Chat with 🦋 حب(2).txt"]
data = prepare_chatbot_data(file_paths)
vectorizer, vectors, chatbot_data = train_chatbot(data)

# Telegram bot token
TOKEN = "7874371911:AAE-H9SFpu0dILwoad_kWu3103T9JqxnfaA"

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

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
