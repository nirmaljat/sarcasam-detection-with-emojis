'''import re
import emoji 

def extract_emojis(text):
    emojis = re.findall(r'\:(.*?)\:', emoji.demojize(text))
    return emojis

text = '🤔 🙈 me así, bla es se 😌 ds 💕👭👙'
emojis_found = extract_emojis(text)

print(emojis_found)'''


import re
import emoji

def replace_emojis_with_names(text):
    def replace(match):
        emoji_name = match.group(1)
        return emoji_name

    replaced_text = re.sub(r':(.*?):', replace, emoji.demojize(text))
    return replaced_text

text = 'My%20mom%20is%20a%20poledancer😎 😭 🤣 😊'
replaced_text = replace_emojis_with_names(text)
x = replaced_text.replace("_", " ")

print(x)
