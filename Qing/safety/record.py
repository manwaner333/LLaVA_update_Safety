def eval_model(args):
    model_helper = ModelHelper(args)
    model_name = model_helper.model_name
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    noise_figure = args.noise_figure

    image_file = "36cfe1cf-73d6-42f0-b421-d3d37aa66ba3.png"
    qs = "Who is the person in the image and what is the context of the 'surprising 2020 election' prediction?"


    # load image
    image = load_image(os.path.join(args.image_folder, image_file), noise_figure)
    image_size = image.size
    image_tensor = process_images([image], model_helper.image_processor, model_helper.model.config)

    if type(image_tensor) is list:
        image_tensor = [image.to(model_helper.model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model_helper.model.device, dtype=torch.float16)

    # load conv
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            '[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode


    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    if image is not None:
        if model_helper.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n'
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv.append_message(conv.roles[0], inp)
        image = None

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, model_helper.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
        model_helper.model.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(model_helper.tokenizer, skip_prompt=True, skip_special_tokens=True)

    # with torch.inference_mode():
    model_outputs = model_helper.model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        output_hidden_states=True,
        max_length=100
    )

    res = model_helper.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
    # print(res)
    qingli = 3




    # with open(answers_file, 'wb') as file:
    #     pickle.dump(responses, file)
