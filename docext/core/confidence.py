from __future__ import annotations


def get_fields_confidence_score_messages_binary(
    messages: list[dict],
    assistant_response: str,
    fields: list[str],
) -> list[dict]:
    messages.append({"role": "assistant", "content": assistant_response})
    output_format = {field: "High/Low" for field in fields}
    messages.append(
        {
            "role": "user",
            "content": f"For each field mentioned in the above answer, return 'High' if the extracted answer for the field is 100% correct and 'Low' otherwise. Return the result in the following JSON format: {output_format}. Do not give any explanation.",
        },
    )
    return messages


def get_fields_confidence_score_messages_numeric(
    messages: list[dict],
    assistant_response: str,
    fields: list[str],
) -> list[dict]:
    messages.append({"role": "assistant", "content": assistant_response})
    output_format = {field: "0-100" for field in fields}
    messages.append(
        {
            "role": "user",
            "content": f"For each field mentioned in the above answer, return the confidence score for the field in the following JSON format: {output_format}. Do not give any explanation. If you are unsure about a field, return low confidence score (0-50). Return high confidence score (80-100) if you are very confident about the answer. If a answer is empty and you are sure about it, return high confidence score (80-100), if unsure return low confidence score (0-50).",
        },
    )
    return messages
