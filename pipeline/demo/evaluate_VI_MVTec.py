import requests
import torch
import transformers
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import textwrap

sys.path.append("../..")
from otter.modeling_otter import OtterForConditionalGeneration

model = OtterForConditionalGeneration.from_pretrained("/home/data/MIMIC-IT/weights/OTTER-Image-MPT7B/", device_map="auto")
tokenizer = model.text_tokenizer
image_processor = transformers.CLIPImageProcessor()

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png']  # 画像の拡張子リスト
    all_files = sorted(os.listdir(folder_path)) # フォルダ内の全てのファイルを取得
    image_paths = [os.path.join(folder_path, file) for file in all_files if os.path.splitext(file)[1].lower() in image_extensions] # 画像のパスを抽出してリストに格納
    return image_paths

def write_text_file(file_path, text):
    with open(file_path, mode="a") as f:
        f.write(text+"\n")
        
def generate_list_string(items):
    # アンダースコアをスペースに変換
    items = [item.replace('_', ' ') for item in items]
    if len(items) == 1:
        return items[0]
    elif len(items) == 2:
        return f"{items[0]} and {items[1]}"
    else:
        return ", ".join(items[:-1]) + f", and {items[-1]}"
        
def test(folder, sub_folder, GTs, model_name, order=True):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    for sub,gt in zip(sub_folder,GTs):
        folder_name = f'./result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/detective.txt', mode='w') as f:
            f.close()
        with open(f'{folder_name}/non-detective.txt', mode='w') as f:
            f.close()
        
        subfolder_string = generate_list_string(GTs)
        model.text_tokenizer.padding_side = "left"
        sentence = f"{sub} --> {gt}"
        write_text_file(f'{folder_name}/detective.txt',sentence)
        
        """ クエリ：不良品 """
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: NG"
            write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/home/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/home/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            
            # inputs = textwrap.dedent(f"""
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
            # """)
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
            """)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: NG"
            write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/home/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/home/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            
            # inputs = textwrap.dedent(f"""
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>Yes {gt}<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>No None<|endofchunk|>
            #     <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
            # """)
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>No. This {folder__} does not have any defects, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>
            """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/detective.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/detective.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if parsed_output.split(" ")[0].lower()=="yes":
                yesno_count += 1
                if len(parsed_output.split(" ")) > 1:
                    if parsed_output.split(" ")[1].lower()==gt.lower():
                        both_count += 1
            if len(parsed_output.split(" ")) > 1:
                if parsed_output.split(" ")[1].lower()==gt.lower():
                        reason_count += 1
                    
            write_text_file(f'{folder_name}/detective.txt',query_image_path)
            write_text_file(f'{folder_name}/detective.txt',parsed_output)
            write_text_file(f'{folder_name}/detective.txt',"")
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"

        write_text_file(f'{folder_name}/detective.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/detective.txt',yesno_acc)
        write_text_file(f'{folder_name}/detective.txt',reason_acc)
        write_text_file(f'{folder_name}/detective.txt',both_acc)
        
        
        """ クエリ：良品 """
        sentence = f"{sub} --> {gt}"
        write_text_file(f'{folder_name}/non-detective.txt',sentence)
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: OK"
            write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: OK"
            write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/non-detective.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/good"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if parsed_output.split(" ")[0].lower()=="no":
                yesno_count += 1
                if len(parsed_output.split(" ")) > 1:
                    if parsed_output.split(" ")[1].lower()=="none":
                        both_count += 1
            if len(parsed_output.split(" ")) > 1:
                if parsed_output.split(" ")[1].lower()=="none":
                        reason_count += 1
                
            write_text_file(f'{folder_name}/non-detective.txt',query_image_path)
            write_text_file(f'{folder_name}/non-detective.txt',parsed_output)
            write_text_file(f'{folder_name}/non-detective.txt',"")
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
        
        write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/non-detective.txt',yesno_acc)
        write_text_file(f'{folder_name}/non-detective.txt',reason_acc)
        write_text_file(f'{folder_name}/non-detective.txt',both_acc)
        
def test_without_context(folder, sub_folder, GTs, model_name):
    # load check point
    trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)
    
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    
    for j, (sub,gt) in enumerate(zip(sub_folder,GTs)):
        folder_name = f'./result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/MVTec_without_context.txt', mode='w') as f:
            f.close()
        
        model.text_tokenizer.padding_side = "left"
        
        sentence = f"{sub} --> {gt}"
        # print(sentence)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',sentence)
        # inputs = textwrap.dedent(f"""
        #    <image>User: This is an image of {folder__}. Does this image have any defects? If there are any defects, please provide the defect name. If not, please say None. GPT:<answer>
        # """)
        inputs = textwrap.dedent(f"""
           <image>User: This is an image of {folder__}. Does this {folder__} have any defects? GPT:<answer>
        """)

        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'{inputs}')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if sub == "good":
                if "no"==parsed_output.split(" ")[0].lower():
                    yesno_count += 1
                    if len(parsed_output.split(" ")) > 1:
                        if gt.lower()==parsed_output.split(" ")[1].lower():
                            both_count += 1
            else:
                if "yes"==parsed_output.split(" ")[0].lower():
                    yesno_count += 1
                    if len(parsed_output.split(" ")) > 1:
                        if gt.lower()==parsed_output.split(" ")[1].lower():
                            both_count += 1
            
            if len(parsed_output.split(" ")) > 1:
                if gt.lower()==parsed_output.split(" ")[1].lower():
                        reason_count += 1
                    
            write_text_file(f'{folder_name}/MVTec_without_context.txt',query_image_path)
            write_text_file(f'{folder_name}/MVTec_without_context.txt',parsed_output)
            write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
        
        yesno_accuracy = f"yesno correct: {yesno_count}, total: {len(query_image_paths)}, acc: {(yesno_count / len(query_image_paths)) * 100:.2f}%"
        reason_accuracy = f"reason correct: {reason_count}, total: {len(query_image_paths)}, acc: {(reason_count / len(query_image_paths)) * 100:.2f}%"
        both_accuracy = f"both correct: {both_count}, total: {len(query_image_paths)}, acc: {(both_count / len(query_image_paths)) * 100:.2f}%"
        
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',yesno_accuracy)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',reason_accuracy)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',both_accuracy)


# 1 run.sh
# model_name = "VI_full_otter_short/batch128_epoch1_lr-5_true"
# model_name = "VI_batch128_short_pairs25"

# folder = "bottle"
# sub_folder = ["broken_large","broken_small","contamination"]
# GTs = ['broken bottle','broken bottle','bottle with contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ['bent','swapp','crack','crack','missing','missing','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination","thread"]
# GTs = ['stain','cut','hole','contamination','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue","metal_contamination","thread"]
# GTs = ["bent","broken","contamination","contamination","contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","glue","poke"]
# GTs = ['stain','scratch','wrinkle','stain','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['stain','contamination','crack','misprint','scratch','stain']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['strip','chip','chip','chip','chip']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['crack','contamination','stain','stain','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","liquid","hole"]
# GTs = ["stain","scratch","stain","hole"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["broken","tear","frayed","frayed","misshapen","misshapen"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)


# 2 run.sh
# model_name = "AC_VI_full_otter_short/batch128_epoch1_lr-5_true"

# folder = "bottle"
# sub_folder = ["broken_large","broken_small","contamination"]
# GTs = ['broken','broken','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ['bent','swapp','crack','crack','missing','missing','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination","thread"]
# GTs = ['stain','cut','hole','contamination','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue","metal_contamination","thread"]
# GTs = ["bent","broken","contamination","contamination","contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","glue","poke"]
# GTs = ['stain','scratch','wrinkle','stain','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['stain','contamination','crack','misprint','scratch','stain']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['strip','chip','chip','chip','chip']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['crack','contamination','stain','stain','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","liquid","hole"]
# GTs = ["stain","scratch","stain","hole"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["broken","tear","frayed","frayed","misshapen","misshapen"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)


def test(folder, sub_folder, GTs, model_name, order=True):
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    for sub,gt in zip(sub_folder,GTs):
        folder_name = f'./result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/detective.txt', mode='w') as f:
            f.close()
        with open(f'{folder_name}/non-detective.txt', mode='w') as f:
            f.close()
        
        subfolder_string = generate_list_string(GTs)
        model.text_tokenizer.padding_side = "left"
        sentence = f"{sub} --> {gt}"
        write_text_file(f'{folder_name}/detective.txt',sentence)
        
        """ クエリ：不良品 """
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: NG"
            write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/home/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/home/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
            """)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: NG"
            write_text_file(f'{folder_name}/detective.txt',sentence)
            demo_image_one = Image.open(f"/home/dataset/mvtec/{folder}/test/{sub}/000.png").resize((224, 224)).convert("RGB")
            demo_image_two = Image.open(f"/home/dataset/mvtec/{folder}/test/good/000.png").resize((224, 224)).convert("RGB")
            
            inputs = textwrap.dedent(f"""
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>Yes. This {folder__} has some {gt}, so it is defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>No. This {folder__} does not have any defects such as {subfolder_string}, so it is non-defective.<|endofchunk|>
                <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
            """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/detective.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/detective.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if parsed_output.split(".")[0].lower()=="yes":
                yesno_count += 1
                    
            write_text_file(f'{folder_name}/detective.txt',query_image_path)
            write_text_file(f'{folder_name}/detective.txt',parsed_output)
            write_text_file(f'{folder_name}/detective.txt',"")
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"

        write_text_file(f'{folder_name}/detective.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/detective.txt',yesno_acc)
        write_text_file(f'{folder_name}/detective.txt',reason_acc)
        write_text_file(f'{folder_name}/detective.txt',both_acc)
        
        
        """ クエリ：良品 """
        sentence = f"{sub} --> {gt}"
        write_text_file(f'{folder_name}/non-detective.txt',sentence)
        if order: # demo_image_one: 良品, demo_image_two: 不良品
            sentence = f"context1: OK, context2: NG, query: OK"
            write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        else: # demo_image_one: 不良品, demo_image_two: 良品
            sentence = f"context1: NG, context2: OK, query: OK"
            write_text_file(f'{folder_name}/non-detective.txt',sentence)
        
        write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/non-detective.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/good"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([demo_image_one, demo_image_two, query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if parsed_output.split(".")[0].lower()=="no":
                yesno_count += 1
                
            write_text_file(f'{folder_name}/non-detective.txt',query_image_path)
            write_text_file(f'{folder_name}/non-detective.txt',parsed_output)
            write_text_file(f'{folder_name}/non-detective.txt',"")
            
        yesno_acc = f"correct: {yesno_count}, total: {len(query_image_paths)-1}, yesno acc: {(yesno_count / (len(query_image_paths)-1)) * 100:.2f}%"
        reason_acc = f"correct: {reason_count}, total: {len(query_image_paths)-1}, reason acc: {(reason_count / (len(query_image_paths)-1)) * 100:.2f}%"
        both_acc = f"correct: {both_count}, total: {len(query_image_paths)-1}, both acc: {(both_count / (len(query_image_paths)-1)) * 100:.2f}%"
        
        write_text_file(f'{folder_name}/non-detective.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/non-detective.txt',yesno_acc)
        write_text_file(f'{folder_name}/non-detective.txt',reason_acc)
        write_text_file(f'{folder_name}/non-detective.txt',both_acc)
        
def test_without_context(folder, sub_folder, GTs, model_name):
    # load check point
    trained_ckpt_path = f'../../log/{model_name}/final_weights.pt'

    train_ckpt = torch.load(trained_ckpt_path, map_location="cpu")
    if train_ckpt.get("model_state_dict", None) is not None:
        train_ckpt = train_ckpt["model_state_dict"]
    _ = model.load_state_dict(train_ckpt, strict=False)
    
    if folder=="grid":
        folder__ = "metal grid"
    else:
        folder__ = folder
    folder__ = folder__.replace('_', ' ')
    subfolder_string = generate_list_string(GTs)
    for j, (sub,gt) in enumerate(zip(sub_folder,GTs)):
        folder_name = f'./result/{folder}/{sub}/{model_name}'
        os.makedirs(folder_name, exist_ok=True)
        with open(f'{folder_name}/MVTec_without_context.txt', mode='w') as f:
            f.close()
        
        model.text_tokenizer.padding_side = "left"
        
        sentence = f"{sub} --> {gt}"
        # print(sentence)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',sentence)
        inputs = textwrap.dedent(f"""
           <image>User: This is an image of {folder__}. Does this {folder__} have any defects such as {subfolder_string}? GPT:<answer>
        """)
        
        inputs = "".join(inputs.split("\n"))
        lang_x = model.text_tokenizer(
            [
                inputs
            ],
            return_tensors="pt",
        )
        
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'-----{sub} start-----')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'{inputs}')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
            
        query_folder_path = f"/home/dataset/mvtec/{folder}/test/{sub}"
        query_image_paths = get_image_paths(query_folder_path)
        yesno_count = 0
        reason_count = 0
        both_count = 0
        for i, query_image_path in enumerate(query_image_paths[1:]):
            # print(query_image_path)
            query_image = Image.open(query_image_path).resize((224, 224)).convert("RGB")
            vision_x = image_processor.preprocess([query_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
        
            # Get the data type from model's parameters
            model_dtype = next(model.parameters()).dtype

            # Convert tensors to the model's data type
            vision_x = vision_x.to(dtype=model_dtype)
            lang_x_input_ids = lang_x["input_ids"]
            lang_x_attention_mask = lang_x["attention_mask"]

            bad_words_id = model.text_tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
            generated_text = model.generate(
                vision_x=vision_x.to(model.device),
                lang_x=lang_x_input_ids.to(model.device),
                attention_mask=lang_x_attention_mask.to(model.device),
                max_new_tokens=512,
                num_beams=3,
                no_repeat_ngram_size=3,
                bad_words_ids=bad_words_id,
            )

            parsed_output = (
                model.text_tokenizer.decode(generated_text[0]).split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
            )
            
            if sub == "good":
                if "no"==parsed_output.split(".")[0].lower():
                    yesno_count += 1
                    
            else:
                if "yes"==parsed_output.split(".")[0].lower():
                    yesno_count += 1
                    
            write_text_file(f'{folder_name}/MVTec_without_context.txt',query_image_path)
            write_text_file(f'{folder_name}/MVTec_without_context.txt',parsed_output)
            write_text_file(f'{folder_name}/MVTec_without_context.txt',"")
        
        yesno_accuracy = f"yesno correct: {yesno_count}, total: {len(query_image_paths)}, acc: {(yesno_count / len(query_image_paths)) * 100:.2f}%"
        reason_accuracy = f"reason correct: {reason_count}, total: {len(query_image_paths)}, acc: {(reason_count / len(query_image_paths)) * 100:.2f}%"
        both_accuracy = f"both correct: {both_count}, total: {len(query_image_paths)}, acc: {(both_count / len(query_image_paths)) * 100:.2f}%"
        
        write_text_file(f'{folder_name}/MVTec_without_context.txt',f'-----{sub} end-----')
        write_text_file(f'{folder_name}/MVTec_without_context.txt',yesno_accuracy)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',reason_accuracy)
        write_text_file(f'{folder_name}/MVTec_without_context.txt',both_accuracy)
        
        
# 1 run2.sh
model_name = "VI_batch128_long_pairs25"

folder = "bottle"
sub_folder = ["broken_large","contamination"]
GTs = ['broken bottle','bottle with contamination']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "cable"
sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","missing_cable","poke_insulation"]
GTs = ['bented cable','swapped cable','cracked cable','missing cable','cable with holes']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "capsule"
sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
GTs = ['cracked capsule','misprinted capsule','capsule with holes','scratched capsule','misshapen capsule']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "carpet"
sub_folder = ["color","cut","hole","metal_contamination"]
GTs = ['stained carpet','scratced carpet','carpet with holes','carpet with contamination']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "grid"
sub_folder = ["bent","broken","glue"]
GTs = ["bent metal grid","broken metal grid","metal grid with contamination"]
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "hazelnut"
sub_folder = ["crack","cut","hole","print"]
GTs = ['cracked hazelnut','scratched hazelnut','hazelnut with holes','misprinted hazelnut']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "leather"
sub_folder = ["color","cut","fold","poke"]
GTs = ['stained leather','scratched leather','wrinkle leather','leather with holes']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "metal_nut"
sub_folder = ["bent","color","flip","scratch"]
GTs = ['bent metal nut','stained metal nut','flipped metal nut','scratched metal nut']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "pill"
sub_folder = ["color","contamination","crack","faulty_imprint","scratch"]
GTs = ['stained pill','pill with contamination','cracked pill','misprinted pill','scratched pill']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "screw"
sub_folder = ["manipulated_front","scratch_head"]
GTs = ['stripped screw','chipped screw']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "tile"
sub_folder = ["crack","glue_strip","gray_stroke"]
GTs = ['cracked tile','tile with contamination','stained tile']
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "toothbrush"
sub_folder = ["defective"]
GTs = ["broken toothbrush"]
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "transistor"
sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
GTs = ["bent transistor","cutted transistor","broken transistor","misalignment transistor"]
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "wood"
sub_folder = ["color","scratch","hole"]
GTs = ["stained wood","scratched wood","wood with holes"]
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

folder = "zipper"
sub_folder = ["broken_teeth","fabric_border","fabric_interior","split_teeth"]
GTs = ["broken zipper","tear zipper","frayed zipper","misshapen zipper"]
test(folder, sub_folder, GTs, model_name)
test_without_context(folder, sub_folder, GTs, model_name)

# folder = "bottle"
# sub_folder = ["broken_large","contamination"]
# GTs = ['broken','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","missing_cable","poke_insulation"]
# GTs = ['bent','swapp','crack','missing','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination"]
# GTs = ['stain','cut','hole','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue"]
# GTs = ["bent","broken","contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","poke"]
# GTs = ['stain','scratch','wrinkle','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch"]
# GTs = ['stain','contamination','crack','misprint','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head"]
# GTs = ['strip','chip']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke"]
# GTs = ['crack','contamination','stain']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","hole"]
# GTs = ["stain","scratch","hole"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","split_teeth"]
# GTs = ["broken","tear","frayed","misshapen"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)


# 2 run2.sh
# model_name = "VI_full_otter_long/batch64_epoch1_lr-5_true"

# folder = "bottle"
# sub_folder = ["broken_large","contamination"]
# GTs = ['broken bottle','bottle with contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","missing_cable","poke_insulation"]
# GTs = ['bented cable','swapped cable','cracked cable','missing cable','cable with holes']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['cracked capsule','misprinted capsule','capsule with holes','scratched capsule','misshapen capsule']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination"]
# GTs = ['stained carpet','scratced carpet','carpet with holes','carpet with contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue"]
# GTs = ["bent metal grid","broken metal grid","metal grid with contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['cracked hazelnut','scratched hazelnut','hazelnut with holes','misprinted hazelnut']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","poke"]
# GTs = ['stained leather','scratched leather','wrinkle leather','leather with holes']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent metal nut','stained metal nut','flipped metal nut','scratched metal nut']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch"]
# GTs = ['stained pill','pill with contamination','cracked pill','misprinted pill','scratched pill']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head"]
# GTs = ['stripped screw','chipped screw']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke"]
# GTs = ['cracked tile','tile with contamination','stained tile']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken toothbrush"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent transistor","cutted transistor","broken transistor","misalignment transistor"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","hole"]
# GTs = ["stained wood","scratched wood","wood with holes"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","split_teeth"]
# GTs = ["broken zipper","tear zipper","frayed zipper","misshapen zipper"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)


############################################################################################################
# # 4
# model_name = "AC_VI_full_otter/batch128_epoch1_lr-5_true"

# folder = "bottle"
# sub_folder = ["broken_large","broken_small","contamination"]
# GTs = ['broken','broken','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ['bent','swapp','crack','crack','missing','missing','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination","thread"]
# GTs = ['stain','cut','hole','contamination','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue","metal_contamination","thread"]
# GTs = ["bent","broken","contamination","contamination","contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","glue","poke"]
# GTs = ['stain','scratch','wrinkle','stain','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['stain','contamination','crack','misprint','scratch','stain']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['strip','chip','chip','chip','chip']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['crack','contamination','stain','stain','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","liquid","hole"]
# GTs = ["stain","scratch","stain","hole"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["broken","tear","frayed","frayed","misshapen","misshapen"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)


# # 5
# model_name = "AC_VI_full_otter/batch128_epoch1_lr-5_false"

# folder = "bottle"
# sub_folder = ["broken_large","broken_small","contamination"]
# GTs = ['broken','broken','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "cable"
# sub_folder = ["bent_wire","cable_swap","cut_inner_insulation","cut_outer_insulation","missing_cable","missing_wire","poke_insulation"]
# GTs = ['bent','swapp','crack','crack','missing','missing','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "capsule"
# sub_folder = ["crack","faulty_imprint","poke","scratch","squeeze"]
# GTs = ['crack','misprint','hole','scratch','misshapen']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "carpet"
# sub_folder = ["color","cut","hole","metal_contamination","thread"]
# GTs = ['stain','cut','hole','contamination','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "grid"
# sub_folder = ["bent","broken","glue","metal_contamination","thread"]
# GTs = ["bent","broken","contamination","contamination","contamination"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "hazelnut"
# sub_folder = ["crack","cut","hole","print"]
# GTs = ['crack','scratch','hole','misprint']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "leather"
# sub_folder = ["color","cut","fold","glue","poke"]
# GTs = ['stain','scratch','wrinkle','stain','hole']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "metal_nut"
# sub_folder = ["bent","color","flip","scratch"]
# GTs = ['bent','stain','flip','scratch']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "pill"
# sub_folder = ["color","contamination","crack","faulty_imprint","scratch","pill_type"]
# GTs = ['stain','contamination','crack','misprint','scratch','stain']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "screw"
# sub_folder = ["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"]
# GTs = ['strip','chip','chip','chip','chip']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "tile"
# sub_folder = ["crack","glue_strip","gray_stroke","oil","rough"]
# GTs = ['crack','contamination','stain','stain','contamination']
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "toothbrush"
# sub_folder = ["defective"]
# GTs = ["broken"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "transistor"
# sub_folder = ["bent_lead","cut_lead","damaged_case","misplaced"]
# GTs = ["bent","cut","broken","misalignment"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "wood"
# sub_folder = ["color","scratch","liquid","hole"]
# GTs = ["stain","scratch","stain","hole"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)

# folder = "zipper"
# sub_folder = ["broken_teeth","fabric_border","fabric_interior","rough","split_teeth","squeezed_teeth"]
# GTs = ["broken","tear","frayed","frayed","misshapen","misshapen"]
# test(folder, sub_folder, GTs, model_name)
# test_without_context(folder, sub_folder, GTs, model_name)