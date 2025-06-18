import re
import textstat


#Lexical requirement smells: 
req_smell_terms = [
    # NASA ARM Tool
    "adequate", "be able to", "timely", "as appropriate", "can", "may", "optionally",
    # QuARS
    "possibly", "eventually", "having in mind", "take into account", "significant",
    # RCM
    "appropriate", "normal", "similar", "better", "worse",
    # ISO 29148 / IEEE 830
    "user friendly", "easy to use", "cost effective", "almost always", "minimal", "if possible", "as applicable", "it"
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
def flesch_reading_ease(text):
    return round(textstat.flesch_reading_ease(text), 2)
