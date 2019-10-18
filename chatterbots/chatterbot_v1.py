

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer,TwitterTrainer
from chatterbot import ChatBot
#from settings import TWITTER
import logging
import pandas as pd


# This example demonstrates how you can train your chat bot
# using data from Twitter.
#
# To use this example, create a new file called settings.py.
# In settings.py define the following:

#使用在线语料库进行训练
# TWITTER = {
#     "CONSUMER_KEY": "my-twitter-consumer-key",
#     "CONSUMER_SECRET": "my-twitter-consumer-secret",
#     "ACCESS_TOKEN": "my-access-token",
#     "ACCESS_TOKEN_SECRET": "my-access-token-secret"}
#
# # Comment out the following line to disable verbose logging
# logging.basicConfig(level=logging.INFO)
#
# chatbot = ChatBot(
#     "TwitterBot",
#     logic_adapters=[
#         "chatterbot.logic.BestMatch"
#     ],
#     input_adapter="chatterbot.input.TerminalAdapter",
#     output_adapter="chatterbot.output.TerminalAdapter",
#     database="./twitter-database.db",
#     twitter_consumer_key=TWITTER["CONSUMER_KEY"],
#     twitter_consumer_secret=TWITTER["CONSUMER_SECRET"],
#     twitter_access_token_key=TWITTER["ACCESS_TOKEN"],
#     twitter_access_token_secret=TWITTER["ACCESS_TOKEN_SECRET"],
#     trainer="chatterbot.trainers.TwitterTrainer")
#
# chatbot.train()
# chatbot.logger.info('Trained database generated successfully!')
# print(chatbot.get_response('hello'))

conversation=pd.read_csv('/Users/admin/Downloads/coca-samples-sources.xlsx','texts')
# print(conversation)

chatbot = ChatBot("new bot")
# conversation = [
#     "Hello",
#     "Hi there!",
#     "How are you doing?",
#     "I'm doing great.",
#     "That is good to hear",
#     "Thank you.",
#     "You're welcome.",
#     "I'm fine."]

chatbot.set_trainer(ListTrainer)
chatbot.train(conversation)

print('')
print(chatbot.get_response('e'))
print(chatbot.get_response('p'))




# bot = ChatBot(
#     "Terminal",
#     storage_adapter="chatterbot.storage.SQLStorageAdapter",
#     logic_adapters=[
#         "chatterbot.logic.MathematicalEvaluation",
#         "chatterbot.logic.TimeLogicAdapter",
#         "chatterbot.logic.BestMatch"
#     ],
#     input_adapter="chatterbot.input.TerminalAdapter",
#     output_adapter="chatterbot.output.TerminalAdapter",
#     database="../database.db"
# )
#
# print("Type something to begin...")
#
# # The following loop will execute each time the user enters input
# while True:
#     try:
#         # We pass None to this method because the parameter
#         # is not used by the TerminalAdapter
#         bot_input = bot.get_response(None)
#
#     # Press ctrl-c or ctrl-d on the keyboard to exit
#     except (KeyboardInterrupt, EOFError, SystemExit):
#         break