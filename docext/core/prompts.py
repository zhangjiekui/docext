from __future__ import annotations

from docext.core.utils import encode_image


def _get_field_desc_prompt(fields, fields_description):
    return "\n".join(
        [
            f"{field.replace(' ', '_').lower()}: {description}"
            for field, description in zip(fields, fields_description)
        ],
    )


def _get_fields_output_format(fields):
    return {field.replace(" ", "_").lower(): "..." for field in fields}


def get_fields_messages(fields, fields_description, filepaths):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract the following fields from the documents:\n {_get_field_desc_prompt(fields, fields_description)}.",
                },
                {"type": "text", "text": f"Documents:\n"},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(filepath)}",
                        },
                    }
                    for filepath in filepaths
                ],
                {
                    "type": "text",
                    "text": f"Return a JSON with the following format:\n {_get_fields_output_format(fields)}. If a field is not found, return '' for that field. Do not give any explanation.",
                },
            ],
        },
    ]
    return messages
