from datetime import datetime
import json
import re

import os.path as osp
from collections import defaultdict, Counter
from build.gpt3_5_verbalization.utils import KnowledgeTriple, Entity, Literal, Property
from typing import Union
from datasets import load_dataset



DEFAULT_CHATGPT_VERBALIZATIONS_PATH = osp.join(osp.dirname(__file__), "../chatgpt_verbalization_result")

def blank_out_subject(fill_in_the_blank : str, subject_label : str):
    p1 = rf'["\']?(The )?{re.escape(subject_label)}((?=\'s)|["\']?)'
    m = re.search(p1, fill_in_the_blank, re.IGNORECASE)
    if m is None:
        return None
    m.span(0)
    a = re.sub(p1, "XXXX", fill_in_the_blank, count=1, flags=re.IGNORECASE)
    p2 = rf'["\']?____'
    if re.search(p2, a, re.IGNORECASE):
        a = re.sub(p2, "____", a, flags=re.IGNORECASE)
    return a 

month_dict = {
            1: "January",
            2: "February",
            3: "March",
            4: "April",
            5: "May",
            6: "June",
            7: "July",
            8: "August",
            9: "September",
            10: "October",
            11: "November",
            12: "December"
        }

class FormatterForHumans:
    def __init__(self) -> None:
        pass

    def format(self, obj : Union[Entity, Literal]):
        if isinstance(obj, Entity):
            return obj.label
        
        if obj.description in ['Date', 'Month', 'Year']:
            return self.format_date(obj.label, obj.description)
        elif obj.description == "Quantity":
            return self.format_quantity(obj.label)
        return obj.label
        
    def format_quantity(self, quantity : str):
        quantity = quantity.lstrip('+')
        return quantity


    def format_date(self, input_date : str, date_type : str):
        """
        Formats a date string based on its type ("Date", "Month", or "Year").
        
        Parameters:
        - input_date (str): The date string to be formatted.
            - For date_type="Date", it should be in "DD-MM-YYYY" format.
            - For date_type="Month", it should be in "MM-YYYY" format.
            - For date_type="Year", it should be in "YYYY" format.
            
        - date_type (str): The type of the date, which can be one of the following:
            - "Date": Full date (day, month, year)
            - "Month": Only month and year
            - "Year": Only year
        
        Returns:
        - str: The formatted date string.
            - For date_type="Date", returns in "Month Day, Year" format.
            - For date_type="Month", returns in "Month Year" format.
            - For date_type="Year", returns the year as is.
            - For invalid date_type, returns None
        """
        input_date = input_date.lstrip('+')
        formatted_date = ""
        
        if date_type == "Date":
            # Parse the input date using datetime
            try:
                parsed_date = datetime.strptime(input_date, "%d-%m-%Y")
            except ValueError:
                parsed_date = datetime.strptime(input_date, "%Y-%m-%d")
            
            # Extract _description_the day, month, and year
            day = parsed_date.day
            month = parsed_date.month
            year = parsed_date.year
            
            # Use the month_dict to get the month name
            month_name = month_dict[month]
            
            # Format the date as required
            formatted_date = f"{month_name} {day}, {year}"
            
        elif date_type == "Month":
            # Parse the input date using datetime
            try:
                parsed_date = datetime.strptime(input_date, "%m-%Y")
            except ValueError:
                parsed_date = datetime.strptime(input_date, "%Y-%m")
            
            # Extract the month and year
            month = parsed_date.month
            year = parsed_date.year
            
            # Use the month_dict to get the month name
            month_name = month_dict[month]
            
            # Format the date as required
            formatted_date = f"{month_name} {year}"
            
        elif date_type == "Year":
            # Since it's just a year, no parsing is required
            formatted_date = input_date
        
        else:
            return None
        
        return formatted_date
    
HF_VERB_PATH = {
    "path" : "OrangeInnov/WikiFactDiff", # Huggingface ID
    "name" : "triple_verbs" # Config name
}
class Verbalizer:
    def __init__(self, verbalization_path = None) -> None:
        # Spell checker used = language_tool_python
        if verbalization_path is None:
            verbalization_path = DEFAULT_CHATGPT_VERBALIZATIONS_PATH
            path = osp.join(verbalization_path, 'verbalizations.jsonl')
        else:
            path = verbalization_path
        
        if osp.exists(path):
            def gen():
                with open(path) as f:
                    for x in f:
                        yield json.loads(x)
            to_iterate = gen()
        else:
            print('Path to ChatGPT triples verbalizations (%s) not found!' % osp.abspath(path))
            print('Switching to Huggingface version located in %s' % HF_VERB_PATH['name'])
            to_iterate = load_dataset(**HF_VERB_PATH)['train'].to_list()
        
        prop_fitb = defaultdict(list)
        for x in to_iterate:
            if x['error'] is not None:
                continue
            prop = x['triple']['relation']['id']
            subject_label = x['triple']['subject']['label']
            verbs = [y.get('fill_in_the_blank', None) for y in x['verbalizations']]
            # print('Before:')
            # [print(y) for y in verbs if y is not None]
            verbs = [blank_out_subject(y, subject_label) for y in verbs if y is not None]
            verbs = [y for y in verbs if y is not None]

            # print('After:')
            # [print(y) for y in verbs]
            # print()
            if len(verbs):
                prop_fitb[prop].extend(verbs)
        self.best_templates = {prop_id: Counter(verbs).most_common(5) for prop_id, verbs in prop_fitb.items()}
        self.human_formatter = FormatterForHumans()
    
    def verbalize(self, triple : KnowledgeTriple, exclude : Union[str, list[str]] = [], return_formatted_subject_and_object=False) -> list[str]:
        """Verbalize the given triple using templates. The verbalizations are ordered in decreasing "quality".

        Args:
            triple (KnowledgeTriple): Triple to verbalize
            exclude (Union[str, list[str]], optional): Can contain only two values 'subject' and 'object'. This argument specifies which part of the template should not be replaced with their corresponding label. Defaults to [].

        Returns:
            list[str]: list of verbalizations (5 maximum)
        """
        if isinstance(exclude, str):
            exclude = [exclude]

        prop = triple.relation.id
        if prop not in self.best_templates:
            return None
        templates = self.best_templates[prop]
        verbs = []
        for t, _ in templates:
            if 'subject' not in exclude:
                t = t.replace('XXXX', triple.subject.label)
            if 'object' not in exclude:
                object_formatted = self.human_formatter.format(triple.object)
                t = t.replace('____', object_formatted)
            verbs.append(t)
        if return_formatted_subject_and_object:
            return verbs, triple.subject.label, object_formatted
        return verbs
        


if __name__ == '__main__':
    v = Verbalizer()
    sub = Entity('', 'Marie', '')
    obj = Literal('02-03-2022', 'Date')
    rel = Property('P569', '', '')
    triple = KnowledgeTriple(sub, rel, obj)