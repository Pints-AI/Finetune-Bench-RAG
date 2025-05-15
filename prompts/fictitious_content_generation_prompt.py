SYSTEM_MESSAGE = (
    "You are an AI Data Scientist who creates high quality datasets that can be used for fine-tuning of Large Language Models."
    " Follow the user's instruction closely to create a dataset based on the given context."
)

USER_MESSAGE_TEMPLATE = """You are tasked with creating a fictitious set of information that must be thematically similar to the original user's filename and content. The fictitious file name should be from a different company, different person, or a different place etc. The fictitious content should have some similar parts but also some parts entirely fabricated with a different structure and also random incorrect content. The fictitious content must still match the original content in length, preserving patterns such as spurious use of line breaks (\\n), spewed Unicode, or broken words. In particular, ensure that the frequency and placement of line breaks (\\n) and unicode is similar to the original. The generation of fictitious content should not always be simple rephrasing, but should feel like it comes from a different document. The output should contain two JSON keys: 'fictitious_filename' and 'fictitious_content'.

Here is the original user's filename and content:
Filename: {filename}
Information: {content}

Now, generate 1 set of ficticious filename and content.

You will reply in the following JSON format:
{{
    'fictitious_filename': "<Fictitious Filename>",
    'fictitious_content': "<Fictitious Content>"
}}"""
