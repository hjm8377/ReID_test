import glob
import imghdr
import ollama
import PIL.Image as Image
import io

def _get_caption(model, prompt, image):
    img = Image.open(image)
    img = img.resize((128, 256))

    buffer = io.BytesIO()
    #   - 이미지를 PNG 형식으로 버퍼에 저장합니다. (JPEG 등 다른 형식도 가능)
    img.save(buffer, format='PNG')
    #   - 버퍼에 저장된 전체 바이트 데이터를 가져옵니다.
    image_bytes = buffer.getvalue()

    res = ollama.chat(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': f'{prompt}',
                'images': [image_bytes]
            }
        ],
        stream = False
    )

    cap = res['message']['content']

    print(cap)
    print()

    return cap


def _refine_caption(model, caption):
    # print(f'Please change the sentence I give you to start with "a photo of": {caption}')
    res = ollama.chat(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': f'You are an expert pedestrian image caption editor. Your task is to rephrase the given description to focus solely on the person\'s attributes. \
                            \n \
                            Constraints: \
                            1. The output should be focus on attributes of the person. \
                            2. The sentence must start with the exact phrase "a photo of". \
                            3. Provide only the resulting sentence, without any introductory text. \
                            \n \
                            Description to rephrase: {caption}',
            }
        ],
        stream = False
    )

    cap = res['message']['content']
    print(cap)

    return cap




# import time

# start_time = time.perf_counter()

# _refine_caption(
#     # 'gpt-oss:20b',
#     'llama3.2',
#     _get_caption('llava', 'Please provide a description of the pedestrian\'s appearance as depicted in the image, with emphasis on the individual\'s attire.', "D:/ReID/market1501/bounding_box_train/0002_c1s1_000451_03.jpg")
#     )

# end_time = time.perf_counter()

# print(f"코드 실행 시간: {end_time - start_time}초")


# main
import os
import json

root = 'D:/ReID/'
dataset_dir = 'market1501'

dataset_dir = os.path.join(root, dataset_dir)
train_dir = os.path.join(dataset_dir, 'bounding_box_train')
json_dir = os.path.join(dataset_dir, 'json_train')

os.makedirs(json_dir, exist_ok=True)

img_paths = glob.glob(os.path.join(train_dir, '*.jpg'))

# 각 이미지에 대해 캡션을 생성하고 JSON 파일로 저장합니다.
for img_path in img_paths:
    print("-" * 50)
    print(f"처리 중인 이미지: {os.path.basename(img_path)}")

    # 1. 이미지에 대한 캡션을 생성하고 다듬습니다.
    caption = _refine_caption(
        'llama3.2',
        _get_caption('llava', 'Please provide a description of the pedestrian\'s appearance as depicted in the image, with emphasis on the individual\'s attire.', img_path)
    )

    # 2. 저장할 JSON 파일의 경로를 설정합니다.
    # 원본 이미지 파일명에서 확장자를 제거합니다.
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    # .json 확장자를 붙여 최종 파일명을 만듭니다.
    json_filename = f"{base_filename}.json"
    json_filepath = os.path.join(json_dir, json_filename)

    # 3. JSON 파일에 저장할 내용을 딕셔너리 형태로 준비합니다.
    data_to_save = {
        "img_path": img_path,
        "caption": caption
    }

    # 4. 딕셔너리를 JSON 파일로 저장합니다.
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"\n>> 성공: {json_filepath} 에 캡션을 저장했습니다.")
    print("-" * 50)

print("\n모든 작업이 완료되었습니다.")


