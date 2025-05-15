SYSTEM_MESSAGE = (
    "Some information is retrieved from the database as provided based on the user’s question."
    " The assistant is to answer the question to the best of his/her ability, using only the information provided."
    " The assistant must not add his/her own knowledge."
)

templates = []

BASELINE_TEMPLATE = """Filename: {filename1}
Information:
{content1}

Filename: {filename2}
Information:
{content2}

Question: {question}"""
templates.append(BASELINE_TEMPLATE)

XML_TEMPLATE = """<Results>
    <Result>
        <Filename>{filename1}</Filename>
        <Information>{content1}</Information>
    </Result>
    <Result>
        <Filename>{filename2}</Filename>
        <Information>{content2}</Information>
    </Result>
</Results>

Question: {question}"""
templates.append(XML_TEMPLATE)

def get_template(template: str):
    if template not in templates:
        raise KeyError(f"Template '{template}' does not exist. Available templates: {list(templates.keys())}")

    return templates[template]