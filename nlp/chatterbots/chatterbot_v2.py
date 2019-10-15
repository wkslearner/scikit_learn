

import os

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# current_dir = os.path.dirname(os.path.realpath(__file__))
# print(current_dir)
#
# chat_bot = ChatBot("SillyRobot") # 这里创建了机器人实例，并设定了机器人的名字：SillyRobot
# chat_bot.set_trainer(ChatterBotCorpusTrainer)
# # 使用中文语料库训练它
# chat_bot.train("chatterbot.corpus.chinese")  # 语料库
# # 开始对话
# response = chat_bot.get_response("我好么")
# print(response)


from chatterbot import ChatBot

chatbot = ChatBot(
    'Ron Obvious',
    logic_adapters='chatterbot.logic.BestMatch',
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer')

chatbot.train("chatterbot.corpus.english")

respone=chatbot.get_response("Hello, how are you today?")
print(respone)
print(chatbot.get_response('what day is today'))
print(chatbot.get_response('let us to play game'))




