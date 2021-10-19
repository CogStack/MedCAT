def encode_examples(left_context, center_context, right_context, tokenizer, max_seq_len):
    left_encoded = tokenizer(left_context)
    center_encoded = tokenizer(center_context)
    right_encoded = tokenizer(right_context)

    input_ids = left_encoded['input_ids']
    attention_mask = left_encoded['attention_mask']
    center_positions = [min(len(x), max_seq_len) for x in input_ids]

    for encoding in [center_encoded, right_encoded]:
        for i in range(len(input_ids)): input_ids[i].extend(encoding['input_ids'][i])
        for i in range(len(input_ids)): attention_mask[i].extend(encoding['attention_mask'][i])

    return {'input_ids': input_ids, 'center_positions': center_positions, 'attention_mask': attention_mask}
