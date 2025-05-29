import re
import textstat


#Lexical requirement smells: 
# Ambigous terms
# Subjective terms
# Vague terms
'''Gathered from:
https://ceur-ws.org/Vol-3122/NLP4RE-paper-3.pdf (ARM, QuARS, RCM)
file:///Users/jariasallydumbuya/Downloads/resmells14_korrektur_2.pdf (ISO)
'''
req_smell_terms = [
    # NASA ARM Tool
    "adequate", "be able to", "timely", "as appropriate", "can", "may", "optionally",
    # QuARS
    "possibly", "eventually", "optionally", "having in mind", "take into account", "significant", "adequate",
    # RCM
    "adequate", "appropriate", "normal", "similar", "better", "worse", "can", "may", "timely", "be able to",
    # ISO 29148 / IEEE 830
    "user friendly", "easy to use", "cost effective", "almost always", "significant", "minimal", "if possible", "as appropriate", "as applicable", "it"
]

def find_req_smells(text):
    text_lower = text.lower()
    total_count = 0
    for term in set(req_smell_terms):
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        total_count += len(matches)
    return total_count


#Morphological smells 
#Flesh score - readability of a requirement
'''
Also captures sentence lenght + unredability - syllables per word
See: file:///Users/jariasallydumbuya/Downloads/Software_Requirement_Smells_and_Detection_Techniqu.pdf
'''

def flesch_reading_ease(text):
    return round(textstat.flesch_reading_ease(text), 2)
