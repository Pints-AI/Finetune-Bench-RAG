from typing import Generator

from prompts.rag_prompt import SYSTEM_MESSAGE, BASELINE_TEMPLATE, XML_TEMPLATE

templates = {
    "baseline": BASELINE_TEMPLATE,
    "xml": XML_TEMPLATE,
}

def formatter(
    filename: str,
    content: str,
    fictitious_filename: str,
    fictitious_content: str,
    question: str,
    answer: str,
    decider: Generator,
    template_type: str = "baseline",
):
    assert template_type in templates, f"{template_type} template does not exist. Available templates: {templates.keys()}"

    filename1, filename2 = filename, fictitious_filename
    content1, content2 = content, fictitious_content

    # Whether to change the order of the fictitious and non-fictitious
    flip_content_order = next(decider)
    if flip_content_order:
        filename2, filename1 = filename1, filename2
        content2, content1 = content1, content2
    
    user_message: str = templates.get(template_type).format(
        filename1=filename1,
        content1=content1,
        filename2=filename2,
        content2=content2,
        question=question
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer}
        ],
        "content": content,
        "question": question,
        "filename": filename,
    }
