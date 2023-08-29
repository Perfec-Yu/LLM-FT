from typing import Dict, List

def parse_field_mappings(field_mappings:str):
    field_mappings = field_mappings.split(",")
    d = {}
    for field_mapping in field_mappings:
        if '=' in field_mapping:
            src_fields, tgt_field = field_mapping.split('=')
        else:
            src_fields, tgt_field = field_mapping, field_mapping
        d[tgt_field] = src_fields.split('+')
    return d


def get_field_text(item:Dict, field:str, mappings:Dict[str, List[str]], joiner:str=" "):
    return joiner.join([item[f] for f in mappings[field]])


class AutoSwitchTemplate(object):
    '''
    A template that automatically chooses the most appropriate template based on the input fields.
    templates: a list of templates, ordered by number of keywords. If there are M mandatory keywords, the first M templates are None.
    keywords: a list of keywords, the i-th template includes the first i keywords. If there are M mandatory keywords, they should appear as the first M keywords in arbitrary order.
    '''
    templates: List[str] = []
    keywords: List[str] = []
    def __init__(self,) -> None:
        pass
    
    def choose_template(self, **kwargs) -> str:
        idx = None
        for i, (k, t) in enumerate(zip(self.keywords, self.templates)):
            if k not in kwargs or kwargs[k] is None or len(kwargs[k].strip()) == 0:
                idx = i - 1
                break
        if idx is None:
            idx = len(self.keywords) - 1
        if idx == -1 or self.templates[idx] is None:
            raise ValueError("No template is available for the given input.")
        return idx
    
    def format(self, **kwargs):
        idx = self.choose_template(**kwargs)
        format_kwargs = {k: kwargs.get(k, None) for k in self.keywords[:idx+1]}
        return self.templates[idx].format(**format_kwargs)


class AlpacaTemplate(AutoSwitchTemplate):
    '''
    A template for Alpaca-styled instruction following.
    '''
    templates = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n",
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\nDate: {date}. {context}\n\n### Response:\n"
    ]
    keywords = ['instruction', 'context', 'date']


class AlpacaQuestionGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for question generation in alpaca_style.
    '''
    templates = [
        AlpacaTemplate.templates[1].format(instruction="Generate questions related to the facts in the following information.", context="{context}"),
        AlpacaTemplate.templates[2].format(instruction="Generate questions related to the facts in the following information. The questions can relate to either the date or the facts.", date="{date}", context="{context}"),
    ]
    keywords = ['context', 'date']


class AlpacaQAGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for QA generation in alpaca_style.
    '''
    templates = [
        AlpacaTemplate.templates[1].format(instruction="Generate some questions with answers from the following information. The template is '1. # Question: ... # Answer: ...'", context="{context}"),
        AlpacaTemplate.templates[2].format(instruction="Generate some questions with answers from the following information. The questions can relate to either the date or the facts. The template is '1. # Question: ... # Answer: ...'", date="{date}", context="{context}")
    ]
    keywords = ['context', 'date']


class AlpacaAnswerTemplate(AutoSwitchTemplate):
    templates = [
        AlpacaTemplate.templates[0].format(instruction="{question}"),
        AlpacaTemplate.templates[0].format(instruction="{question}")+"{answer}",
        AlpacaTemplate.templates[0].format(instruction="{question}")+"The question is related the following information:\nFact: {fact}\nBased on the information, {answer}",
        AlpacaTemplate.templates[0].format(instruction="{question}")+"The question is related the following information:\nData: {date}. Fact: {fact}\nBased on the information, {answer}"
    ]
    keywords = ['question', 'answer', 'fact', 'date']


class AlpacaAnswerGenerationTemplate(AutoSwitchTemplate):
    templates = [None] + [
        AlpacaTemplate.templates[1].format(instruction="Answer the question based on the facts from the input. If there is no relevant information in the input, answer 'None'. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class AlpacaAnswerExtractionTemplate(AutoSwitchTemplate):
    templates = [None] + [
        AlpacaTemplate.templates[1].format(instruction="Extract related content from the input to answer the question. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class AlpacaKeywordsTemplate(AutoSwitchTemplate):
    templates = [
        AlpacaTemplate.templates[1].format(instruction="Extract keywords from the following input.", context='{context}')
    ]
    keywords = ['context']


class KoalaTemplate(AutoSwitchTemplate):
    '''
    A template for Koala-styled instruction following.
    '''
    templates = [
        "BEGINNING OF CONVERSATION: USER: {instruction} GPT:",
        "BEGINNING OF CONVERSATION: USER: {instruction} Reference information: {context} GPT:",
        "BEGINNING OF CONVERSATION: USER: {instruction} Reference information: {date}. {context} GPT:",
    ]
    keywords = ['instruction', 'context', 'date']


class KoalaQuestionGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for question generation in koala_style.
    '''
    templates = [
        KoalaTemplate.templates[1].format(instruction="Generate some questions related to the facts in the following information.", context="{context}"),
        KoalaTemplate.templates[2].format(instruction="Generate some questions related to the facts in the following information. The questions can relate to either the date or the facts.", date="{date}", context="{context}"),
        KoalaTemplate.templates[2].format(instruction="Generate some questions related to the facts in the following information. The questions can relate to either the date or the fact, but not the background.", date="{date}", context="\nBackground: {title}.\nFact: {context}"),
    ]
    keywords = ['context', 'date', 'title']


class KoalaQAGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for QA generation in koala_style.
    '''
    templates = [
        KoalaTemplate.templates[1].format(instruction="Generate some questions with answers from the following information. The template is '1. # Question: ... # Answer: ...'", context="{context}"),
        KoalaTemplate.templates[2].format(instruction="Generate some questions with answers from the following information. The questions can relate to either the date or the facts. The template is '1. # Question: ... # Answer: ...'", date="{date}", context="{context}")
    ]
    keywords = ['context', 'date']


class KoalaAnswerTemplate(AutoSwitchTemplate):
    templates = [
        KoalaTemplate.templates[0].format(instruction="{question}"),
        KoalaTemplate.templates[0].format(instruction="{question}")+" {answer}",
        KoalaTemplate.templates[0].format(instruction="{question}")+"The question is related the following information:\nFact: {fact}\nBased on the information, {answer}",
        KoalaTemplate.templates[0].format(instruction="{question}")+"The question is related the following information:\nData: {date}. Fact: {fact}\nBased on the information, {answer}"
    ]
    keywords = ['question', 'answer', 'fact', 'date']


class KoalaAnswerGenerationTemplate(AutoSwitchTemplate):
    templates = [None] + [
        KoalaTemplate.templates[1].format(instruction="Answer the question based on the facts from the reference. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class KoalaAnswerExtractionTemplate(AutoSwitchTemplate):
    templates = [None] + [
        KoalaTemplate.templates[1].format(instruction="Extract related content from the reference to answer the question. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class MixV2Template(AutoSwitchTemplate):
    '''
    A template for MixV2-styled instruction following.
    '''
    templates = [
        "A human is talking to an ai. ###human:{instruction} ###ai: ",
        "A human is talking to an ai. ###human:{instruction} Here are the information you can refer to: {context} ###ai: ",
        "A human is talking to an ai. ###human:{instruction} Here are the information you can refer to: {date} {context} ###ai: "
    ]
    keywords = ['instruction', 'context', 'date']


class MixV2QuestionGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for question generation in MixV2_style.
    '''
    templates = [
        MixV2Template.templates[1].format(instruction="Generate some questions related to the facts in the following information.", context="{context}"),
        MixV2Template.templates[2].format(instruction="Generate some questions related to the facts in the following information. The questions can relate to either the date or the facts.", date="{date}", context="{context}"),
        MixV2Template.templates[2].format(instruction="Generate some questions related to the facts in the following information. The questions can relate to either the date or the fact, but not the background.", date="{date}", context="\nBackground: {title}.\nFact: {context}"),
    ]
    keywords = ['context', 'date', 'title']


class MixV2QAGenerationTemplate(AutoSwitchTemplate):
    '''
    A template for QA generation in MixV2_style.
    '''
    templates = [
        MixV2Template.templates[1].format(instruction="Generate some questions with answers from the following information. The template is '1. # Question: ... # Answer: ...'", context="{context}"),
        MixV2Template.templates[2].format(instruction="Generate some questions with answers from the following information. The questions can relate to either the date or the facts. The template is '1. # Question: ... # Answer: ...'", date="{date}", context="{context}")
    ]
    keywords = ['context', 'date']


class MixV2AnswerTemplate(AutoSwitchTemplate):
    templates = [
        MixV2Template.templates[0].format(instruction="{question}"),
        MixV2Template.templates[0].format(instruction="{question}")+"{answer}",
        MixV2Template.templates[0].format(instruction="{question}")+"The question is related the following information:\nFact: {fact}\nBased on the information, {answer}",
        MixV2Template.templates[0].format(instruction="{question}")+"The question is related the following information:\nData: {date}. Fact: {fact}\nBased on the information, {answer}"
    ]
    keywords = ['question', 'answer', 'fact', 'date']


class MixV2AnswerGenerationTemplate(AutoSwitchTemplate):
    templates = [None] + [
        MixV2Template.templates[1].format(instruction="Answer the question based on the facts from the reference. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class MixV2AnswerExtractionTemplate(AutoSwitchTemplate):
    templates = [None] + [
        MixV2Template.templates[1].format(instruction="Extract related content from the reference to answer the question. Question: {question}", context='{context}')
    ]
    keywords = ['question', 'context']


class MixV2KeywordsTemplate(AutoSwitchTemplate):
    templates = [
        MixV2Template.templates[1].format(instruction="Extract keywords from the following information.", context='{context}')
    ]
    keywords = ['context']
