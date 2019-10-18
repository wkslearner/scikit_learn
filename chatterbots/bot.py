
from chatterbot import ChatBot

# Create a new chat bot named Charlie
chatbot = ChatBot(
    'Ron Obvious',
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer')


# Get a response to the input text 'How are you?'
print(chatbot.get_response('what ？'))
print(chatbot.get_response('so stupit'))
print(chatbot.get_response('how are you'))
print(chatbot.get_response('how to be sucessful'))
print(chatbot.get_response('i am not sure who you are'))
print(chatbot.get_response('please,do not say what'))





