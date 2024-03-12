import json
from html import escape
import random

from build.verbalize_wikifactdiff.verbalize_wikifactdiff import SAVE_PATH
import os.path as osp

def json2html(json_dict : dict):
    # Start HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>JSON Visualizer</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
            }
            .container {
                width: 50%;
                margin: auto;
            }
            .collapsible {
                background-color: #2c3e50;
                color: white;
                cursor: pointer;
                padding: 10px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 15px;
                margin-top: 5px;
                border-radius: 5px;
            }
            .active, .collapsible:hover {
                background-color: #1a2533;
            }
            .content {
                padding: 0 18px;
                display: none;
                overflow: hidden;
                background-color: #eaeaea;
                border-radius: 5px;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin-bottom: 5px;
            }
        </style>
    </head>
    <body>
    <div class="container">
    """

    # JavaScript for collapsible content
    html_content += """
    <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            var coll = document.getElementsByClassName('collapsible');
            for (var i = 0; i < coll.length; i++) {
                coll[i].addEventListener('click', function() {
                    this.classList.toggle('active');
                    var content = this.nextElementSibling;
                    if (content.style.display === 'block') {
                        content.style.display = 'none';
                    } else {
                        content.style.display = 'block';
                    }
                });
            }
        });
    </script>
    """

    def generate_html(key, value):
        html = f"<button class='collapsible'>{escape(str(key))}</button><div class='content'>"
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                html += generate_html(sub_key, sub_value)
        elif isinstance(value, list):
            html += "<ul>"
            for index, item in enumerate(value):
                if isinstance(item, (list, dict)):
                    html += generate_html(index, item)
                else:
                    html += f"<li>{escape(str(item))}</li>"
            html += "</ul>"
        else:
            html += f"<p>{escape(str(value))}</p>"
        html += "</div>"
        return html

    # Check if the root of JSON data is a list or a dictionary
    if isinstance(json_dict, list):
        html_content += generate_html('Root', json_dict)
    elif isinstance(json_dict, dict):
        for key, value in json_dict.items():
            html_content += generate_html(key, value)

    # End HTML content
    html_content += """
    </div>
    </body>
    </html>
    """
    return html_content

dataset = [json.loads(x) for x in open(SAVE_PATH)]
sample = random.sample(dataset, k=20)

with open(osp.join(osp.dirname(SAVE_PATH), 'visualize_wfd_sample.html'), 'w') as f:
    f.write(json2html(sample))
