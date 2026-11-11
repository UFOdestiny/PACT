"""Render the frozen math datasets into the shared training conversation."""
from tools.response_protocol import chat_messages

def process_dataset(dataset, script_args):
    def conversation(example):
        problem = example['question'] if 'question' in example else example['problem']
        return {'prompt': chat_messages(script_args.sys_prompt_name, "math", problem)}
    return dataset.map(conversation)
