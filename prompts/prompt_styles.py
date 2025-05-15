import warnings
from abc import abstractmethod
from typing import Dict, List, Type, Union


class PromptStyle:
    """Base interface for prompt styles."""

    @abstractmethod
    def apply(self, prompt: str, **kwargs: str) -> str:
        raise NotImplementedError('PromptStyle.apply() is an abstract method.')

    @classmethod
    def from_name(cls, name: str) -> 'PromptStyle':
        if name not in prompt_styles:
            return None
        return prompt_styles[name]()


# This default style is exported from the original open_instruct repository:
# https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py#L396-L407
class Default(PromptStyle):
    def apply(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        **kwargs: str,
    ) -> str:
        message_text = ''
        for message in prompt:
            if message['role'] == 'system':
                message_text += '<|system|>\n' + message['content'].strip() + '\n'
            elif message['role'] == 'user':
                message_text += '<|user|>\n' + message['content'].strip() + '\n'
            elif message['role'] == 'assistant':
                message_text += (
                    '<|assistant|>\n' + message['content'].strip() + '</s>' + '\n'
                )
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        return message_text


class Llama3_1(PromptStyle):
    DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

    def apply(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        no_system: bool = False,
        append_assistant_header: bool = False,
        **kwargs: str,
    ) -> str:
        assert isinstance(
            prompt, list
        ), f'Unsupported prompt type: {type(prompt)}. prompt should be formatted in a list of dict<role: str, content: str>'

        tokens = []
        if not no_system and not self.has_system_prompt(prompt):
            tokens.extend(
                self.encode_message(
                    {'role': 'system', 'content': Llama3_1.DEFAULT_SYSTEM_MESSAGE}
                )
            )

        for i, message in enumerate(prompt):
            if i != 0 and message['role'] == 'system':
                raise ValueError("'system' role is only allowed at the beginning of the conversation list.")
            if message['role'] not in ['assistant', 'user', 'system']:
                warnings.warn(
                    f"Unknown role: '{message['role']}'. Supported roles are 'assistant', 'user', and 'system'. It is assumed that this is intended.",
                    UserWarning,
                )

            tokens.extend(self.encode_message(message))

        if append_assistant_header:
            tokens.extend(self.encode_header('assistant'))

        return ''.join(tokens)

    def encode_header(self, role: str) -> List[str]:
        return [f'<|start_header_id|>{role}<|end_header_id|>\n\n']

    def encode_message(self, message: Dict[str, str]) -> List[str]:
        tokens = self.encode_header(message['role'])
        # NOTE: Meta stripped this. I'm not sure I agree, but who am I to argue?
        tokens.append(message['content'].strip())
        tokens.append('<|eot_id|>')
        return tokens

    def has_system_prompt(self, messages: List[Dict[str, str]]) -> bool:
        return messages[0].get('role', '') == 'system' if len(messages) else False


class OLMoE(PromptStyle):
    def apply(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        add_generation_prompt: bool = False,
        bos_token: str = '<s>',
        eos_token: str = '</s>',
        **kwargs,
    ) -> str:
        assert isinstance(
            prompt, list
        ), f"Expected prompt to be a list of messages, got {type(prompt)}"

        result = [bos_token]
        for i, message in enumerate(prompt):
            role = message.get("role")
            content = message.get("content", "").strip()

            if role == "system":
                result.append("<|system|>\n" + content)
            elif role == "user":
                result.append("<|user|>\n" + content)
            elif role == "assistant":
                result.append("<|assistant|>\n" + content + eos_token)
            else:
                raise ValueError(f"Unsupported message role: {role}")

            # Append assistant header if it's the last message and generation is expected
            if i == len(prompt) - 1 and add_generation_prompt:
                result.append("<|assistant|>")

        return "\n".join(result) + "\n"


prompt_styles: Dict[str, Type[PromptStyle]] = {
    'default': Default,
    'llama3.1': Llama3_1,
    'olmoe': OLMoE,
}
